import os
from dataclasses import dataclass
import shutil
from learning_loop_node import loop, ModelInformation
import asyncio
from glob import glob
import json
from yolov5_trainer import Yolov5Trainer
from learning_loop_node.trainer import Trainer
from learning_loop_node.context import Context
from learning_loop_node.rest.downloader import DataDownloader
from learning_loop_node.rest.downloads import download_model
from fastapi.encoders import jsonable_encoder
from typing import List


@dataclass
class Project:
    project_id: str
    organization_id: str

    def __init__(self, project_id, organization_id):
        self.project_id = project_id
        self.organization_id = organization_id

    @staticmethod
    def from_dict(project: dict):
        return Project(project['project_id'], project['organization_id'])


def create_model_info(file: dict) -> ModelInformation:
    data = json.load(file)
    return ModelInformation(id=data['id'], host=loop.base_url, organization=data['organization'], project=data['project'], version=data['version'], categories=data['categories'], resolution=data['resolution'])


class ContinousDetector:
    def __init__(self, loop):
        self.loop = loop
        self.projects = []
        self.data_path = '/tmp'

    async def get_projects(self):
        async with loop.get('api/projects') as response:
            assert response.status == 200, response
            self.projects = [Project.from_dict(project) for project in (await response.json())['projects']]

    async def get_deployment_target(self, project_id: str, organization_id: str) -> str:
        async with loop.get(f'/api/{organization_id}/projects/{project_id}/deployment/target') as response:
            assert response.status == 200, response
            return (await response.json())['target_id']

    async def get_image_ids_without_detections(self, project_id: str, organization_id: str, deployment_target: str) -> List[str]:
        image_ids = []
        for state in ['inbox', 'annotate', 'review', 'complete']:
            async with loop.get(f'/api/{organization_id}/projects/{project_id}/data?detections=true&model_version={deployment_target}&state={state}') as response:
                assert response.status == 200, response
                new_ids = (await response.json())['image_ids']
                image_ids += new_ids
        return image_ids

    async def get_model_id_from_deployment_target(self, organization: str, project: str, deployment_target: str) -> str:
        async with loop.get(f'/api/{organization}/projects/{project}/models') as response:
            models = (await response.json())['models']
            target_model_id = [model['id'] for model in models if model['version'] == deployment_target]
            if target_model_id:
                return target_model_id[0]
            return ''


async def main():
    cd = ContinousDetector(loop)
    await cd.get_projects()
    for project in cd.projects:
        print(f'Current project: {project.project_id}')
        context = Context(organization=project.organization_id, project=project.project_id)
        deployment_target = await cd.get_deployment_target(project.project_id, project.organization_id)
        if deployment_target is not None:
            print(f'Using deployment target {deployment_target}')
            ids = await cd.get_image_ids_without_detections(project.project_id, project.organization_id, deployment_target)
            image_folder = f'{cd.data_path}/images/{project.organization_id}/{project.project_id}'
            await DataDownloader(context).download_images(ids, image_folder)
            images = [img for img in glob(f'{image_folder}/**/*.*', recursive=True)
                      if os.path.splitext(os.path.basename(img))[0] in ids]

            model_folder = f'{cd.data_path}/models/{project.organization_id}/{project.project_id}'
            shutil.rmtree(model_folder, ignore_errors=True)
            os.makedirs(model_folder, exist_ok=True)
            model_id = await cd.get_model_id_from_deployment_target(project.organization_id, project.project_id, deployment_target)
            if model_id:
                await download_model(model_folder, context, model_id, 'yolov5_pytorch')
                with open(f'{model_folder}/model.json') as m:
                    model_info = create_model_info(m)

                print("Start detecting")
                detections = await Yolov5Trainer()._detect(model_info, images, model_folder)
                await Trainer('yolov5_pytorch')._upload_detections(context, jsonable_encoder(detections))
            else:
                print(f'Could not download model for project {context.project} in format yolov5_pytorch')


if __name__ == "__main__":
    start = input('Start detect? [y/n]')
    if start == 'y':
        while True:
            loop.download_token()
            asyncio.run(main())
            print('Finished')

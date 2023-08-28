import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from glob import glob
from typing import List

from fastapi.encoders import jsonable_encoder
from learning_loop_node import GLOBALS, ModelInformation, loop
from learning_loop_node.context import Context
from learning_loop_node.rest.downloader import DataDownloader
from learning_loop_node.rest.downloads import download_model
from learning_loop_node.trainer import Trainer

from yolov5_trainer import Yolov5Trainer

logging.basicConfig(level=logging.INFO)


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


class ContinousDetector:
    def __init__(self, loop):
        self.loop = loop
        self.projects = []
        self.data_path = GLOBALS.data_folder

    async def get_projects(self):
        async with loop.get('api/projects') as response:
            assert response.status == 200, response
            self.projects = [Project.from_dict(project) for project in (await response.json())['projects']]

    async def get_deployment_target(self, project_id: str, organization_id: str) -> str:
        async with loop.get(f'/api/{organization_id}/projects/{project_id}/deployment/target') as response:
            assert response.status == 200, response
            return (await response.json())['target_id']

    async def get_image_ids_without_detections(self, project_id: str, organization_id: str) -> List[str]:
        async with loop.get(f'/api/{organization_id}/projects/{project_id}/deployment/imageIds') as response:
            assert response.status == 200, response
            return (await response.json())

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
        logging.info(f'Current project: {project.project_id}')
        context = Context(organization=project.organization_id, project=project.project_id)
        ids = await cd.get_image_ids_without_detections(context.project, context.organization)
        if ids:
            logging.info(f'Downloading {len(ids)} images')
            image_folder = f'{cd.data_path}/images/{context.organization}/{context.project}'
            await DataDownloader(context).download_images(ids, image_folder)
            images = [img for img in glob(f'{image_folder}/**/*.*', recursive=True)
                      if os.path.splitext(os.path.basename(img))[0] in ids]

            model_folder = '/tmp/background_detector/models'
            shutil.rmtree(model_folder, ignore_errors=True)
            os.makedirs(model_folder)
            deployment_target = await cd.get_deployment_target(context.project, context.organization)
            model_id = await cd.get_model_id_from_deployment_target(context.organization, context.project, deployment_target)
            if model_id:
                logging.info(f'Downloading model {model_id} (version {deployment_target})')
                await download_model(model_folder, context, model_id, 'yolov5_pytorch')
                model_info = ModelInformation.load_from_disk(model_folder)
                logging.info(f"Start detecting {len(ids)} images")
                detections = await Yolov5Trainer()._detect(model_info, images, model_folder)
                logging.info(f'Uploading {len(detections)} detections')
                await Trainer('yolov5_pytorch')._upload_detections(context, jsonable_encoder(detections))
            else:
                logging.info(f'Could not download model for project {context.project} in format yolov5_pytorch')


if __name__ == "__main__":
    while True:
        loop.download_token()
        asyncio.run(main())
        logging.info('Finished')

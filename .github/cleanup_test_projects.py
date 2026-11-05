"""Delete the trainer test projects CI runs left behind on the shared loop.

The trainer job pipes this into its container as its last step, also after a cancellation. The run's own project
goes first. Any other is deleted once the run named in its suffix is completed or has been re-run since; a run in
progress or one that cannot be looked up keeps its project. A delete that times out ends the sweep.
"""
import asyncio
import logging
import os
import re

import httpx
from learning_loop_node.loop_communication import LoopCommunicator

ORGANIZATION = 'zauberzeug'
# the suffix the workflow sets: <run id>_<run attempt>
SUFFIXED = re.compile(r'^pytest_yolo5det_(?P<run_id>\d+)_(?P<attempt>\d+)$')
# longer than the loop's own 60 s limit, so a slow delete brings back the loop's answer
DELETE_TIMEOUT = 75


async def main() -> None:
    loop_communicator = LoopCommunicator()
    github = httpx.AsyncClient(base_url='https://api.github.com', timeout=30,
                               headers={'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
                                        'Accept': 'application/vnd.github+json'})
    try:
        own = f'pytest_yolo5det_{os.environ["LOOP_TEST_PROJECT_SUFFIX"]}'
        response = await loop_communicator.get('/projects')
        response.raise_for_status()
        listed = {p['project_id'] for p in response.json()['projects']
                  if p['organization_id'] == ORGANIZATION and SUFFIXED.match(p['project_id'])}
        logging.info('%d test projects listed', len(listed))
        for project in [own] + await stale_projects(github, listed - {own}):
            try:
                response = await loop_communicator.delete(f'/{ORGANIZATION}/projects/{project}',
                                                          timeout=DELETE_TIMEOUT)
            except httpx.TimeoutException:
                logging.warning('deleting %s timed out, the remaining projects are left to the next sweep', project)
                break
            logging.info('deleting %s: %s %s', project, response.status_code,
                         response.text if response.status_code not in (200, 404) else '')
    finally:
        await github.aclose()
        await loop_communicator.shutdown()


async def stale_projects(client: httpx.AsyncClient, projects: set[str]) -> list[str]:
    stale = []
    for project in sorted(projects):
        match = SUFFIXED.match(project)
        assert match is not None
        try:
            run = await run_status(client, match['run_id'])
        except httpx.HTTPError as e:
            logging.warning('keeping %s, its run cannot be looked up: %s', project, e)
            continue
        if run['status'] == 'completed' or int(match['attempt']) < run['run_attempt']:
            stale.append(project)
        else:
            logging.info('keeping %s', project)
    return stale


async def run_status(client: httpx.AsyncClient, run_id: str) -> dict:
    response = await client.get(f'/repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run_id}')
    response.raise_for_status()
    return response.json()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    asyncio.run(main())

import os

from learning_loop_node.tests import test_helper

from dotenv import load_dotenv

load_dotenv()

from icecream import ic
import requests



class NewServerSession():

    def __init__(self, *args, **kwargs):
        ic('INIT NEW SERVER SESSION')
        data = {
            'username': (None, os.environ.get('LOOP_USERNAME', None)),
            'password': (None, os.environ.get('LOOP_PASSWORD', None)),
        }
        self.cookies = requests.post('https://preview.learning-loop.ai/api/login', data=data).cookies
        print('post', 'https://preview.learning-loop.ai/api/login')
        ic(self.cookies)


# data = {
#     'username': (None, os.environ.get('LOOP_USERNAME', None)),
#     'password': (None, os.environ.get('LOOP_PASSWORD', None)),
# }
# ic(data)
# cookies = requests.post(f'https://preview.learning-loop.ai/api/login', data=data).cookies
# print('post', f'https://preview.learning-loop.ai/api/login')
# ic(cookies)

new_server_session = NewServerSession()

print('\n-----------')

response = test_helper.LiveServerSession().get(f"/zauberzeug/projects/demo/data")
assert response.status_code == 200
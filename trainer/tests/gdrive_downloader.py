#!/usr/bin/env python3
import requests
from icecream import ic
# https://stackoverflow.com/a/39225272/4082686


def download(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={
        'id': id,
        'confirm': 't'},  # e.g. large file warning.
        stream=True)
    ic(str(response))
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    ic(response.cookies)
    for key, value in response.cookies.items():
        ic(key, value)
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    file_id = '1jNgQDqQeaZhIWFCxV1eKeuLzXHYhXAQv'
    destination = 'test.zip'
    download(file_id, destination)

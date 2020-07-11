import http.client
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request

import requests

# Face API Key and Endpoint
f = open('api_key.txt', 'r')
data = f.read().split("\n")
subscription_key = data[0]
uri_base = data[1]

cloak_image_base = 'http://sandlab.cs.uchicago.edu/fawkes/files/cloak/{}_high_cloaked.png'
original_image_base = 'http://sandlab.cs.uchicago.edu/fawkes/files/cloak/{}.png'


def test_cloak():
    NUM_TRAIN = 5
    total_idx = range(0, 82)
    TRAIN_RANGE = random.sample(total_idx, NUM_TRAIN)

    TEST_RANGE = random.sample([i for i in total_idx if i not in TRAIN_RANGE], 20)

    personGroupId = 'all'

    # delete_personGroup(personGroupId)
    # create_personGroupId(personGroupId, personGroupId)

    with open("protect_personId.txt", 'r') as f:
        protect_personId = f.read()
    print(protect_personId)
    delete_personGroupPerson(personGroupId, protect_personId)

    protect_personId = create_personId(personGroupId, 'Emily')
    with open("protect_personId.txt", 'w') as f:
        f.write(protect_personId)

    print("Created protect personId: {}".format(protect_personId))
    for idx in TRAIN_RANGE:
        image_url = cloak_image_base.format(idx)
        r = add_persistedFaceId(personGroupId, protect_personId, image_url)
        if r is not None:
            print("Added {}".format(idx))
        else:
            print("Unable to add {}-th image of protect person".format(idx))

    # add other people
    # for idx_person in range(5000, 15000):
    #     personId = create_personId(personGroupId, str(idx_person))
    #     print("Created personId: {}".format(idx_person))
    #     for idx_image in range(10):
    #         image_url = "http://sandlab.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(
    #             idx_person, idx_image)
    #         r = add_persistedFaceId(personGroupId, personId, image_url)
    #         if r is not None:
    #             print("Added {}".format(idx_image))
    #         else:
    #             print("Unable to add {}-th image".format(idx_image))

    # train model based on personGroup

    train_personGroup(personGroupId)

    while json.loads(get_trainStatus(personGroupId))['status'] != 'succeeded':
        time.sleep(2)

    # list_personGroupPerson(personGroupId)

    # test original image
    idx_range = TEST_RANGE
    acc = 0.
    tot = 0.
    for idx in idx_range:
        original_image_url = original_image_base.format(idx)
        faceId = detect_face(original_image_url)
        if faceId is None:
            print("{} does not exist".format(idx))
            continue
        original_faceIds = [faceId]

        # verify
        res = eval(original_faceIds, personGroupId, protect_personId)
        if res:
            acc += 1.
        tot += 1.

    acc /= tot
    print(acc)  # 1.0


def list_personGroups():
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("GET", "/face/v1.0/persongroups?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def detect_face(image_url):
    r = requests.get(image_url)
    if r.status_code != 200:
        return None

    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
        # Request parameters
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'recognitionModel': 'recognition_01',
        'returnRecognitionModel': 'false',
        'detectionModel': 'detection_01',
    })

    body = json.dumps({
        'url': image_url
    })

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("POST", "/face/v1.0/detect?%s" % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    #
    # print(data)
    conn.close()
    return data[0]["faceId"]


def verify_face(faceId, personGroupId, personId):
    # html header
    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    # image URL
    body = json.dumps({
        "faceId": faceId,
        "personId": personId,
        "PersonGroupId": personGroupId
    })

    # Call Face API
    conn = http.client.HTTPSConnection(uri_base)
    conn.request("POST", "/face/v1.0/verify?%s" % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    conn.close()
    return data


def create_personGroupId(personGroupId, personGroupName):
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({
        "name": personGroupName
    })

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("PUT", "/face/v1.0/persongroups/{}?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def create_personId(personGroupId, personName):
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({
        "name": personName
    })

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("POST", "/face/v1.0/persongroups/{}/persons?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    # print(data)
    conn.close()
    return data["personId"]


def add_persistedFaceId(personGroupId, personId, image_url):
    r = requests.get(image_url)
    if r.status_code != 200:
        return None

    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
        'personGroupId': personGroupId,
        'personId': personId
    })

    body = json.dumps({
        'url': image_url
    })

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("POST",
                 "/face/v1.0/persongroups/{}/persons/{}/persistedFaces?%s".format(personGroupId, personId) % params,
                 body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    conn.close()
    if "persistedFaceId" not in data:
        return None
    return data["persistedFaceId"]


def list_personGroupPerson(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("GET", "/face/v1.0/persongroups/{}/persons?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    conn.close()
    for person in data:
        print(person["personId"], len(person["persistedFaceIds"]))


def get_personGroupPerson(personGroupId, personId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("GET", "/face/v1.0/persongroups/{}/persons/{}?%s".format(personGroupId, personId) % params, body,
                 headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    print(data)
    conn.close()


def train_personGroup(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("POST", "/face/v1.0/persongroups/{}/train?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def eval(original_faceIds, personGroupId, protect_personId):
    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({
        'faceIds': original_faceIds,
        'personGroupId': personGroupId,
        'maxNumOfCandidatesReturned': 1
    })

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("POST", "/face/v1.0/identify?%s" % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    conn.close()
    face = data[0]
    print(face)
    if len(face["candidates"]) and face["candidates"][0]["personId"] == protect_personId:
        return True
    else:
        return False


def delete_personGroupPerson(personGroupId, personId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("DELETE", "/face/v1.0/persongroups/{}/persons/{}?%s".format(personGroupId, personId) % params, body,
                 headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def get_trainStatus(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({})

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("GET", "/face/v1.0/persongroups/{}/training?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return data


def delete_personGroup(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("DELETE", "/face/v1.0/persongroups/{}?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def main():
    test_cloak()

    # delete_personGroup('cloaking')
    # delete_personGroup('cloaking-emily')
    # delete_personGroup('pubfig')
    # list_personGroups()
    # exit()
    # personGroupId = 'cloaking'
    # create_personGroupId(personGroupId, 'cloaking')
    # delete_personGroupPerson(personGroupId, '0ac606cd-24b3-440f-866a-31adf2a1b446')
    # add_protect_person(personGroupId, 'Emily')
    # protect_personId = create_personId(personGroupId, 'Emily')
    # add_sybil_person(personGroupId, 'sybil')
    #
    # # train model based on personGroup
    # train_personGroup(personGroupId)
    # get_trainStatus(personGroupId)
    # add_other_person(personGroupId)
    # list_personGroupPerson(personGroupId)
    #
    # idx_range = range(72, 82)
    # original_faceIds = []
    # for idx in idx_range:
    #     original_image_url = "https://super.cs.uchicago.edu/~shawn/cloaked/{}_o.png".format(idx)
    #     faceId = detect_face(original_image_url)
    #     original_faceIds.append(faceId)
    #
    #     # verify
    #     eval(original_faceIds, personGroupId, protect_personId)


if __name__ == '__main__':
    test_cloak()

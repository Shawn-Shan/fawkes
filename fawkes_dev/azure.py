
import http.client, urllib.request, urllib.parse, urllib.error
import json
import time

#Face API Key and Endpoint
subscription_key = 'e127e26e4d534e2bad6fd9ca06145302'
uri_base = 'eastus.api.cognitive.microsoft.com'
# uri_base = 'https://shawn.cognitiveservices.azure.com/'

def detect_face(image_url):
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
    print(data)
    conn.close()
    return data["personId"]


def add_persistedFaceId(personGroupId, personId, image_url):
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
    conn.request("POST", "/face/v1.0/persongroups/{}/persons/{}/persistedFaces?%s".format(personGroupId, personId) % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    print(data)
    conn.close()
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
    conn.request("GET", "/face/v1.0/persongroups/{}/persons/{}?%s".format(personGroupId, personId) % params, body, headers)
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
    conn.request("DELETE", "/face/v1.0/persongroups/{}/persons/{}?%s".format(personGroupId, personId) % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def add_protect_person(personGroupId, name):
    personId = create_personId(personGroupId, name)
    for idx in range(72):
        cloaked_image_url = "https://super.cs.uchicago.edu/~shawn/cloaked/{}_c.png".format(idx)
        add_persistedFaceId(personGroupId, personId, cloaked_image_url)


def add_sybil_person(personGroupId, name):
    personId = create_personId(personGroupId, name)
    for idx in range(82):
        try:
            cloaked_image_url = "https://super.cs.uchicago.edu/~shawn/sybils/{}_c.png".format(idx)
            add_persistedFaceId(personGroupId, personId, cloaked_image_url)
        except:
            print(idx)


def add_other_person(personGroupId):
    for idx_person in range(65):
        personId = create_personId(personGroupId, str(idx_person))
        for idx_image in range(90):
            try:
                image_url = "https://super.cs.uchicago.edu/~shawn/train/{}/{}.png".format(idx_person, idx_image)
                add_persistedFaceId(personGroupId, personId, image_url)
            except:
                print(idx_person, idx_image)


def get_trainStatus(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
    })

    body = json.dumps({})

    conn = http.client.HTTPSConnection(uri_base)
    conn.request("GET", "/face/v1.0/persongroups/{}/training?%s".format(personGroupId) % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()


def test_original():
    personGroupId = 'pubfig'
    # create_personGroupId(personGroupId, 'pubfig')
    # add protect person
    protect_personId = 'd3df3012-6f3f-4c1b-b86d-55e91a352e01'
    #protect_personId = create_personId(personGroupId, 'Emily')
    #for idx in range(50):
    #    image_url = "https://super.cs.uchicago.edu/~shawn/cloaked/{}_o.png".format(idx)
    #    add_persistedFaceId(personGroupId, protect_personId, image_url)

    # add other people
    #for idx_person in range(65):
    #    personId = create_personId(personGroupId, str(idx_person))
    #    for idx_image in range(50):
    #        try:
    #            image_url = "https://super.cs.uchicago.edu/~shawn/train/{}/{}.png".format(idx_person, idx_image)
    #            add_persistedFaceId(personGroupId, personId, image_url)
    #        except:
    #            print(idx_person, idx_image)


    # train model based on personGroup
    #train_personGroup(personGroupId)
    #time.sleep(3)
    #get_trainStatus(personGroupId)
    #list_personGroupPerson(personGroupId)

    idx_range = range(50, 82)
    acc = 0.

    for idx in idx_range:
        original_image_url = "https://super.cs.uchicago.edu/~shawn/cloaked/{}_o.png".format(idx)
        faceId = detect_face(original_image_url)
        original_faceIds = [faceId]

        # verify
        res = eval(original_faceIds, personGroupId, protect_personId)
        if res:
            acc += 1.

    acc /= len(idx_range)
    print(acc) # 1.0


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
    # delete_personGroup('cloaking')
    # delete_personGroup('cloaking-emily')
    # delete_personGroup('pubfig')
    # list_personGroups()
    # exit()
    personGroupId = 'cloaking'
    # create_personGroupId(personGroupId, 'cloaking')
    list_personGroups()
    exit()
    #delete_personGroupPerson(personGroupId, '0ac606cd-24b3-440f-866a-31adf2a1b446')
    #add_protect_person(personGroupId, 'Emily')
    #personId = create_personId(personGroupId, 'Emily')
    #add_sybil_person(personGroupId, 'sybil')
    protect_personId = '6c5a71eb-f39a-4570-b3f5-72cca3ab5a6b'
    #delete_personGroupPerson(personGroupId, protect_personId)
    #add_protect_person(personGroupId, 'Emily')

    # train model based on personGroup
    #train_personGroup(personGroupId)
    get_trainStatus(personGroupId)
    #add_other_person(personGroupId)
    #list_personGroupPerson(personGroupId)
    #delete_personGroupPerson(personGroupId, '80e32c80-bc69-416a-9dff-c8d42d7a3301')

    idx_range = range(72, 82)
    original_faceIds = []
    for idx in idx_range:
        original_image_url = "https://super.cs.uchicago.edu/~shawn/cloaked/{}_o.png".format(idx)
        faceId = detect_face(original_image_url)
        original_faceIds.append(faceId)

        # verify
        eval(original_faceIds, personGroupId, protect_personId)


if __name__ == '__main__':
    main()
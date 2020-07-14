import glob
import sys
import time

import azure
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials

test_dir = '/Users/sixiongshan/Desktop/kash_private/'
cloak_mode = "mid"

f = open('api_key.txt', 'r')
data = f.read().split("\n")
KEY = data[0]
ENDPOINT = "https://shawn.cognitiveservices.azure.com/"


def main():
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    PERSON_GROUP_ID = 'testing'

    try:
        face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)
    except azure.cognitiveservices.vision.face.models._models_py3.APIErrorException:
        pass

    # dummy = face_client.person_group_person.create(PERSON_GROUP_ID, "dummy")
    # m = open("foo.png", 'r+b')
    # face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, dummy.person_id, m)

    test_person = face_client.person_group_person.create(PERSON_GROUP_ID, "test")
    try:
        test_images = [file for file in glob.glob(test_dir + '*')]
        test_images = [i for i in test_images if "cloak" not in i]
        l = len(test_images)
        if l == 1:
            raise Exception("Must have 2 images...")
        print("{} test images".format(l))

        training_images = test_images[:l // 2]
        testing_images = test_images[l // 2:]
        print(training_images, test_images)

        for image in training_images:
            format = 'png'
            name = ".".join(image.split(".")[:-1])
            image = name + "_{}_cloaked.".format(cloak_mode) + format
            print(image)
            m = open(image, 'r+b')
            face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, test_person.person_id, m)

        print('Training the person group...')
        # Train the person group
        face_client.person_group.train(PERSON_GROUP_ID)

        while (True):
            training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
            print("Training status: {}.".format(training_status.status))
            print()
            if (training_status.status is TrainingStatusType.succeeded):
                break
            elif (training_status.status is TrainingStatusType.failed):
                sys.exit('Training the person group has failed.')
            time.sleep(2)

        # Detect faces
        face_ids = []
        for img in testing_images:
            print(img)
            img = open(img, 'r+b')
            faces = face_client.face.detect_with_stream(img)
            for face in faces:
                face_ids.append(face.face_id)

        print("Detect {} Faces out of {} images".format(len(face_ids), len(testing_images)))

        results = face_client.face.identify(face_ids, PERSON_GROUP_ID)

        correct = 0
        for person in results:
            if len(person.candidates) == 0:
                continue
            detected_person = person.candidates[0]
            p_id = detected_person.person_id
            conf = detected_person.confidence
            if p_id == test_person.person_id:
                correct += 1
                print("correctly matched with conf {}".format(conf))

        print("Final match success rate: {}".format(float(correct / len(face_ids))))
    except Exception as e:
        print(e)

    face_client.person_group_person.delete(PERSON_GROUP_ID, test_person.person_id)


# def add_other_people():
#     for idx_person in range(0, 5000):
#         personId = create_personId(personGroupId, str(idx_person))
#         print("Created personId: {}".format(idx_person))
#         for idx_image in range(10):
#             image_url = "http://sandlab.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(
#                 idx_person, idx_image)
#             r = add_persistedFaceId(personGroupId, personId, image_url)
#             if r is not None:
#                 print("Added {}".format(idx_image))
#             else:
#                 print("Unable to add {}-th image".format(idx_image))


if __name__ == '__main__':
    main()

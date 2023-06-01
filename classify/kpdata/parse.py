import json
import csv

all_keypoints = []

current_person = []

def parse_json_from_file(file_path):
    global current_person
    with open(file_path, 'r') as f:
        data = json.load(f)
    
        # Parse captures
        captures = data['captures']
        for capture in captures:
            
            # Parse annotations
            annotations = capture['annotations']
            for annotation in annotations:
                # print(f"\nAnnotation ID: {annotation['id']}")
                # print(f"Annotation definition: {annotation['annotation_definition']}")
                # print(f"File name: {annotation.get('filename', 'N/A')}")  # filename might not always be available
                
                # Parse values
                for value in annotation.get('values', []):  # values might not always be available
                    # print(f"\nLabel ID: {value.get('label_id', 'N/A')}")
                    # print(f"Instance ID: {value.get('instance_id', 'N/A')}")
                    # print(f"Template GUID: {value.get('template_guid', 'N/A')}")
                    # print(f"Pose: {value.get('pose', 'N/A')}")
                    
                    # Parse keypoints
                    for keypoint in value.get('keypoints', []):  # keypoints might not always be available
                        current_person.append((keypoint.get('x'),keypoint.get('y')))

                        if len(current_person) == 17:
                            all_keypoints.append(current_person)
                            current_person = []

parse_json_from_file('lying_captures_000.json')
# You can call this function like so:
# parse_json(your_json_string)
# print(all_keypoints[:3])
with open('lying_keypoints.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for person in all_keypoints:
        writer.writerow(person)

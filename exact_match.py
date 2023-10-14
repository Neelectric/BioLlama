import json

with open('output/Llama-2-70B-MedQA_USMLE_train_ALL10178.json', 'rb') as json_file:
    json_data = json_file.read().decode('utf-8')
data = json.loads(json_data)
#print(data[0])

num_correct = 0
num_total = 0
for i in range(len(data)):
    num_total += 1
    if data[i][1] == data[i][2]:
        num_correct += 1

print("Accuracy is " + str(num_correct/num_total) + " out of " + str(num_total) + " questions.")
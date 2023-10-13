import json

with open('output/Llama-2-70B-MedQA_USMLE_train.json', 'rb') as json_file:
    json_data = json_file.read().decode('utf-8')
data = json.loads(json_data)
#print(data[0])

num_correct = 0
num_total = 0
for instance in data:
    num_total += 1
    if ("(" + instance[2]) == instance[1]:
        num_correct += 1

print("percentage is " + str(num_correct/num_total))
print(data[917][2])
for i in range(len(data)):
    data[i][2] = "(" + data[i][2]

print(data[917][2])

# with open("output/Llama-2-70B-MedQA_USMLE_train_processed.json", "w") as outfile: 
#     json.dump(data, outfile)
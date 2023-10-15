import json

output_file = "output/Llama-2-13B-chat-GPTQ-MedQA_USMLE.json"

with open(output_file, 'rb') as json_file:
    json_data = json_file.read().decode('utf-8')
data = json.loads(json_data)
#print(data[0])

num_correct = 0
num_total = 0
for i in range(len(data)):
    num_total += 1
    if data[i][1] == data[i][2]:
        num_correct += 1

print("Accuracy is " + str(num_correct/num_total) + " with a total of " + str(num_total) + " questions.")
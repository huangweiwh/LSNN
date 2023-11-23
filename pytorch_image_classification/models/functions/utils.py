import csv

def create_csv(path, head):
    with open(path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        csv_file.writerow(head)


def append_csv(path, content):
    with open(path, "a+", newline='') as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        csv_file.writerow(content)

def read_csv(path):
    with open(path,"r+") as file:
        csv_file = csv.reader(file)
        for data in csv_file:
            print("data:", data)

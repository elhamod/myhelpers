

import csv
import json
import os
import pickle

class Reader_writer():
    def __init__(self, type_, folder_name, file_name):
        self.type_=type_
        self.folder_name=folder_name
        self.file_name=file_name
        self.full_name = os.path.join(folder_name, file_name)
        
    def writeFile(self, obj):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        # try:
        if self.type_ == "csv":
            with open(self.full_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(obj)
        elif self.type_=="json":
            with open(self.full_name, 'w') as f:
                json.dump(obj, f)
        elif self.type_=="pkl":
            pickle.dump(obj, open(self.full_name, "wb"))
        else:
            print("File type not supported")
            raise

        print('file',self.full_name,'written')
        # except:
        #     print("Couldn't write", self.full_name)
        #     pass
            
    def readFile(self):
        try:
            if self.type_ == "csv":
                with open(self.full_name, newline='') as f:
                    reader = csv.reader(f)
                    loaded = list(reader)
                    print('file',self.full_name,'read')
                    loaded[0] = [int(i) for i in loaded[0]]
                    return loaded[0]
            elif self.type_=="json":
                with open(self.full_name) as f:
                    return json.load(f)
            elif self.type_=="pkl":
                return pickle.load(open(self.full_name, "rb" ) )
            else:
                print("File type not supported")
                raise
        except:
            print("Couldn't read", self.full_name)
            pass  

class CSV_reader_writer(Reader_writer):
    def __init__(self, folder_name, file_name):
        super().__init__("csv", folder_name, file_name)

class JSON_reader_writer(Reader_writer):
    def __init__(self, folder_name, file_name):
        super().__init__("json", folder_name, file_name)


class Pickle_reader_writer(Reader_writer):
    def __init__(self, folder_name, file_name):
        super().__init__("pkl", folder_name, file_name)
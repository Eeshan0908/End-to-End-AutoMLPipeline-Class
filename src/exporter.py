import pickle
import json
import os


class Exporter:

    @staticmethod
    def create_output_folder():

        os.makedirs("output",exist_ok=True)

    @staticmethod
    def save_model(model,file_name="best_model.pkl"):

        Exporter.create_output_folder()

        file_path = os.path.join("output",file_name)

        with open(file_path,"wb") as file:

            pickle.dump(model,file)

        print(f"\nModel saved: {file_path}")

    @staticmethod
    def save_results(results,file_name="model_results.json"):

        Exporter.create_output_folder()

        file_path = os.path.join("output",file_name)

        with open(file_path,"w") as file:

            json.dump(results,file,indent=4)

        print(f"\nResults saved: {file_path}")
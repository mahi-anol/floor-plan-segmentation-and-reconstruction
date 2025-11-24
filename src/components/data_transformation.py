import sys
import re
from xml.etree import ElementTree as ET
from dataclasses import dataclass
import cairosvg
import os
import shutil
from src.exception import CustomException
from src.logger import logging
from tqdm import tqdm
import pickle

@dataclass
class DataTransformationConfig:
    raw_train_data_dir: str="./artifacts/raw-data/train"
    raw_test_data_dir: str="./artifacts/raw-data/test"
    fixed_train_svg_dir: str="./artifacts/fixed-data/train"
    fixed_test_svg_dir: str="./artifacts/fixed-data/test"
    processed_data_base_dir:str = "./artifacts/processed-data"
    processed_train_data_dir:str="./artifacts/processed-data/train"
    processed_test_data_dir:str="./artifacts/processed-data/test"
    class_to_color_mapping_path:str= "./artifacts/processed-data/color_to_class.pkl"

class DataTransformation:
    # Static Variable for class mapping.
    color_to_class_mapping={(0,0,0):'Background'}

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    @staticmethod
    # utility
    def svg_to_png_convert(svg_path:str,image_path:str,fixed_svg_dir:str,processed_data_dir:str,index=None):

        """
            Desc: A function which converts the svg to a png while perserving the original size.
            Args: file_path
            Return: None but generates a json file with color to class mapping.
        """

        # Read the SVG 
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_data = f.read()

        #  Extract width & height from tags 
        width_match = re.search(r"<width>(\d+)</width>", svg_data)
        height_match = re.search(r"<height>(\d+)</height>", svg_data)

        if not (width_match and height_match):
            raise ValueError("Width or height tag not found in SVG!")

        width = int(width_match.group(1))
        height = int(height_match.group(1))

        #  Parse XML and add width/height attributes
        root = ET.fromstring(svg_data)
        root.set("width", str(width))
        root.set("height", str(height))

        # Remove old <width> and <height> elements
        for tag in root.findall("width"):
            root.remove(tag)
        for tag in root.findall("height"):
            root.remove(tag)

        file_name=os.path.basename(svg_path)


        # Write fixed SVG 
        fixed_svg_path = f"{fixed_svg_dir}/{index}/{file_name}"
        fixed_svg_dir=os.path.dirname(fixed_svg_path)
        os.makedirs(fixed_svg_dir,exist_ok=True)

        ET.ElementTree(root).write(fixed_svg_path, encoding="utf-8", xml_declaration=True)

        # Convert to PNG using CairoSVG 
        png_path = f"{processed_data_dir}/{index}/{file_name.split('.')[0]}.png"
        processed_data_dir=os.path.dirname(png_path)
        os.makedirs(processed_data_dir,exist_ok=True)

        cairosvg.svg2png(url=fixed_svg_path, write_to=png_path)
        shutil.copy(image_path,f"{processed_data_dir}")

        # print(f"SVG converted to PNG successfully at {png_path}")
        # Storing color by class mapping 
        for polygon in root.findall(".//{http://www.w3.org/2000/svg}polygon"):
            fill = polygon.get("fill")
            class_name = polygon.get("class")
            if fill not in DataTransformation.color_to_class_mapping:
                hex_color = fill.lstrip("#")
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                DataTransformation.color_to_class_mapping[rgb] = class_name
            
    def initiate_data_transformation(self):

        try:
            logging.info("Starting data transformation...")
            # For train data.
            folders=os.listdir(self.data_transformation_config.raw_train_data_dir)
            for i,folder in enumerate(tqdm(folders,desc="Train data transformation",leave=False)):
                folder_path=os.path.join(self.data_transformation_config.raw_train_data_dir,folder)
                files=os.listdir(folder_path)
                file1=files[0]
                file2=files[1]

                if not file1.endswith('.svg'): # file 1 should be always .svg
                    file1,file2=file2,file1

                svg_path=os.path.join(folder_path,file1).replace('\\','/')
                image_path=os.path.join(folder_path,file2).replace('\\','/')

                self.svg_to_png_convert(svg_path=svg_path,
                                        image_path=image_path,
                                        fixed_svg_dir=self.data_transformation_config.fixed_train_svg_dir,
                                        processed_data_dir=self.data_transformation_config.processed_train_data_dir,
                                        index=i)
            
            # For test data.
            folders=os.listdir(self.data_transformation_config.raw_test_data_dir)
            for i,folder in enumerate(tqdm(folders,desc="Test data transformation",leave=False)):
                folder_path=os.path.join(self.data_transformation_config.raw_test_data_dir,folder)
                files=os.listdir(folder_path)
                file1=files[0]
                file2=files[1]

                if not file1.endswith('.svg'): # file 1 should be always .svg
                    file1,file2=file2,file1

                svg_path=os.path.join(folder_path,file1)
                image_path=os.path.join(folder_path,file2)
                self.svg_to_png_convert(svg_path=svg_path,
                                        image_path=image_path,
                                        fixed_svg_dir=self.data_transformation_config.fixed_test_svg_dir,
                                        processed_data_dir=self.data_transformation_config.processed_test_data_dir,
                                        index=i)
            
            with open(file=DataTransformationConfig.class_to_color_mapping_path,mode='wb') as json_file:
                pickle.dump(DataTransformation.color_to_class_mapping,json_file,protocol=pickle.HIGHEST_PROTOCOL)

            logging.info("Successfully finished data transformation")
            return DataTransformationConfig.processed_train_data_dir,DataTransformationConfig.processed_test_data_dir
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    processed_train_dir,processed_test_dir=DataTransformation().initiate_data_transformation()
    print("processed artifacts at (%s,%s)" %(processed_train_dir,processed_test_dir))
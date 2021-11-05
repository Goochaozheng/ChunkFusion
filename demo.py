import yaml
from easydict import EasyDict
from module import Pipeline

if __name__ == "__main__":

    configFile = open("./configs/demo.yaml", 'r')
    config = EasyDict(yaml.safe_load(configFile))
    
    pipeline = Pipeline(config)
    pipeline.run()

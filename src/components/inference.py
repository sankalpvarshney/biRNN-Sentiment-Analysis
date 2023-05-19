import tensorflow as tf
from src.utils.common import read_yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


class SentimentAnalysis():

    def __init__(self, params) -> None:
        
        self.model_dir = os.path.join(params['train']['artifact_dir'],params['train']['checkpoint_dir'],params['train']['model_dir'])
        self.model = tf.keras.models.load_model("artifact/ckpt/model")



    def predict(self, text):

        output = self.model.predict([text])

        return output[0][0]
    
if __name__ == "__main__":

    params = read_yaml("config/config.yaml")
    obj = SentimentAnalysis(params)
    text = "My experiance towards te acting was good one but movie is short"
    output = obj.predict(text)
    print(f"Score for the input sentence {output}")
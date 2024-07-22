# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('NeuroSync Lipscape')
    st.info('This application is originally developed from the Lip-Reader deep learning model.')

st.title('NeuroSync Lipscape Full Stack App') 

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, '..', 'data', 's1')

# Print the current working directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory: {data_dir}")

# Check if the data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} does not exist. Please check the path.")

# Generating a list of options or videos 
options = os.listdir(data_dir)
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(data_dir, selected_video)
        try:
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        except:
            print("Error in ffmpeg conversion")

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)  

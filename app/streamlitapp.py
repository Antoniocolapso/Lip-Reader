# Import all of the dependencies
import streamlit as st
import os
import imageio
import numpy as np
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
data_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 's1'))

# Print the current working directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory: {data_dir}")

# Check if the data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} does not exist. Please check the path.")

# Generating a list of options or videos 
options = os.listdir(data_dir)
if not options:
    raise FileNotFoundError(f"No video files found in directory {data_dir}. Please check the path.")

selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if selected_video: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(data_dir, selected_video)
        output_video_path = os.path.join(BASE_DIR, 'test_video.mp4')
        ffmpeg_command = f'ffmpeg -i {file_path} -vcodec libx264 {output_video_path} -y'
        
        try:
            os.system(ffmpeg_command)
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"ffmpeg did not create the output file {output_video_path}.")
        except Exception as e:
            st.error(f"Error during ffmpeg conversion: {e}")
            raise e

        # Rendering inside of the app
        with open(output_video_path, 'rb') as video_file:
            video_bytes = video_file.read() 
            st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        
        # Assume alignment path is in the same directory as the video file but with a different extension
        z = os.path.splitext(file_path)[0] 
        l = z.split("/")
        alignment_path = "".join("/" + l[i] for i in range(1, len(l) - 2)) + "/alignments" + "".join("/" + l[i] for i in range(len(l) - 2, len(l))) + ".align"
        
        print(f"Alignment path: {alignment_path}")

        if not os.path.exists(alignment_path):
            raise FileNotFoundError(f"Alignment file {alignment_path} does not exist. Please check the path.")

        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # animation_path = os.path.join(BASE_DIR, 'animation.gif')
        # # Convert video frames to uint8 and clip values
        # video_frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in video]
        # # Ensure frames are in RGB format
        # video_frames_rgb = [np.stack([frame, frame, frame], axis=-1) for frame in video_frames] # Convert grayscale to RGB
        # imageio.mimsave(animation_path, video_frames, fps=10)
        # st.image(animation_path, width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

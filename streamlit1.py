from streamlit_webrtc import webrtc_streamer
import streamlit as st 
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from pathlib import Path

# mediapipe init
mp_hands = mp.solutions.hands

# Load the pre-trained gesture classification model
model = tf.keras.models.load_model('model.h5')  # Substitua pelo caminho para o seu modelo



def main():
    # html_temp = """
    # <div style="background-color:#f4f4f4 ;padding:10px;margin:auto">
    # st.image("1.png")
    # </div>
    # """
    # st.markdown(html_temp, unsafe_allow_html=True)
    st.image("path1.png")

    st.sidebar.title("Menu")
    # Add a selectbox to the sidebar:
    pages=['HANDY']
    add_pages = st.sidebar.selectbox('', pages)

    st.sidebar.title("Criadores:")
    html_temp6 = """
    <ul style="font-weight:bold;">
    <li>Gabriel Dias</li>
    <li>Isabelle Melo</li>
    <li>Lucas Reis </li>
    <li>Gustavo Melo</li>
    </ul>
    """
    st.sidebar.markdown(html_temp6, unsafe_allow_html=True)

    if add_pages=='HANDY':
        html_temp2 = """
    <body style="background-color:#black ;padding:10px;">
    <h3 style="color:white ;text-align:center;">Sobre</h3>
    <p style="text-align:justify;">O trabalho visa a implementação de um classificador de gestos ofensivos utilizando a abordagem de aprendizado profundo. Para isso, foi criado um banco de dados de imagens separadas em duas classes: ofensivo e não ofensivo. Este banco de dados alimentou o treinamento dos modelos, respectivamente separados em redes convolucionais [1] e redes pré-treinadas a partir do Xception [2] com dados coloridos e em escala de cinza. Os resultados finais mostram que modelos de aprendizado profundo, principalmente no contexto de imagens,dependem de um grande volume de dados para que a tarefa de compreensão de padrão de dados
    seja bem sucedida.</p>
    </body>
    <div style="background-color:;padding:10px;margin-bottom:10px;">
    <h4 style="color:white;">Prepared using:</h4>
    <ul style="color:white;">
    <li>Deep Learning </li>
    <li>Processamento de Imagens</li>
    <li>Transfer Learning </li>
    <li>Opencv </li>
    <li>Keras </li>
    <li>Streamlit </li>
    <li>PyAutoGui </li>

    </ul>
    </div>
"""
        st.markdown(html_temp2, unsafe_allow_html=True)

    FRAME_WINDOW1 = st.image([])

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    run = st.button('Start Web Camera')
    stop = st.button('Stop')
    with mp_hands.Hands (
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while run:
                sucess, frame = cap.read()
                if not sucess:
                    break

                # BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        # Hands coordinates
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            h, w, _ = frame.shape
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            landmarks.append((x, y))

                        # Bounding box
                        x_min = min(landmarks, key=lambda x: x[0])[0]
                        x_max = max(landmarks, key=lambda x: x[0])[0]
                        y_min = min(landmarks, key=lambda x: x[1])[1]
                        y_max = max(landmarks, key=lambda x: x[1])[1]

                        # Resize bounding box
                        expansion_factor = 1.5
                        width = x_max - x_min
                        height = y_max - y_min
                        x_min = max(0, x_min - int(width * (expansion_factor - 1) / 2))
                        x_max = min(frame.shape[1], x_max + int(width * (expansion_factor - 1) / 2))
                        y_min = max(0, y_min - int(height * (expansion_factor - 1) / 2))
                        y_max = min(frame.shape[0], y_max + int(height * (expansion_factor - 1) / 2))

                        # Extract hand gesture image
                        hand_gesture = frame[y_min:y_max, x_min:x_max]

                        # Preprocess hand gesture image (resize to model input size)
                        hand_gesture = cv2.resize(hand_gesture, (150, 150))
                        hand_gesture = hand_gesture / 255.0  # Normalize
                        hand_gesture = np.expand_dims(hand_gesture, axis=0)  # Add batch dimension

                        # Classify hand gesture using the model
                        prediction = model.predict(hand_gesture)


                        probability_offensive = prediction[0][0]
                        probability_non_offensive = 1 - probability_offensive

                        # Definir a classe com base na probabilidade
                        if probability_non_offensive > probability_offensive:
                            predicted_class = "Non-offensive"
                        else:
                            predicted_class = "Offensive"

                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Display class label
                        cv2.putText(frame, predicted_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if predicted_class == 'Offensive':
                            # Gaussian blur 
                            roi = frame[y_min:y_max, x_min:x_max]
                            roi = cv2.GaussianBlur(roi, (121, 121), 0)
                            frame[y_min:y_max, x_min:x_max] = roi
                        FRAME_WINDOW1.image(frame)

                # Show resulting image
                # cv2.imshow('Hand Detection', frame)
                    # ret, buffer = cv2.imencode('.jpg', frame)
                    # if not ret:
                    #     break
                    # frame = buffer.tobytes()
                    # yield (b'--frame\r\n'
                    #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    

                    if stop:
                        break
                    

                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






# def video_frame_callback(frame): 
#     # Webcam init
#     cap = cv2.VideoCapture(0)

#     with mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             results = hands.process(rgb_frame)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:

#                     # Hands coordinates
#                     landmarks = []
#                     for landmark in hand_landmarks.landmark:
#                         h, w, _ = frame.shape
#                         x, y = int(landmark.x * w), int(landmark.y * h)
#                         landmarks.append((x, y))

#                     # Bounding box
#                     x_min = min(landmarks, key=lambda x: x[0])[0]
#                     x_max = max(landmarks, key=lambda x: x[0])[0]
#                     y_min = min(landmarks, key=lambda x: x[1])[1]
#                     y_max = max(landmarks, key=lambda x: x[1])[1]

#                     # Resize bounding box
#                     expansion_factor = 1.5
#                     width = x_max - x_min
#                     height = y_max - y_min
#                     x_min = max(0, x_min - int(width * (expansion_factor - 1) / 2))
#                     x_max = min(frame.shape[1], x_max + int(width * (expansion_factor - 1) / 2))
#                     y_min = max(0, y_min - int(height * (expansion_factor - 1) / 2))
#                     y_max = min(frame.shape[0], y_max + int(height * (expansion_factor - 1) / 2))

#                     # Extract hand gesture image
#                     hand_gesture = frame[y_min:y_max, x_min:x_max]

#                     # Preprocess hand gesture image (resize to model input size)
#                     hand_gesture = cv2.resize(hand_gesture, (150, 150))
#                     hand_gesture = hand_gesture / 255.0  # Normalize
#                     hand_gesture = np.expand_dims(hand_gesture, axis=0)  # Add batch dimension

#                     # Classify hand gesture using the model
#                     prediction = model.predict(hand_gesture)


#                     probability_offensive = prediction[0][0]
#                     probability_non_offensive = 1 - probability_offensive

#                     # Definir a classe com base na probabilidade
#                     if probability_non_offensive > probability_offensive:
#                         predicted_class = "Non-offensive"
#                     else:
#                         predicted_class = "Offensive"

#                     # Draw bounding box
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#                     # Display class label
#                     cv2.putText(frame, predicted_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                     if predicted_class == 'Offensive':
#                         # Gaussian blur 
#                         roi = frame[y_min:y_max, x_min:x_max]
#                         roi = cv2.GaussianBlur(roi, (121, 121), 0)
#                         frame[y_min:y_max, x_min:x_max] = roi

#             # Show resulting image
#             cv2.imshow('Hand Detection', frame)
            

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

            
#             return cap

#     cap.release()
#     cv2.destroyAllWindows()

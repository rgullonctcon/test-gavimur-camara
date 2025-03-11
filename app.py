import io
import time

import cv2
import torch
import numpy as np

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

import streamlit as st

from ultralytics import YOLO

from typing import List, Tuple, Dict

from shapely.geometry import Point, Polygon
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from shapely.geometry import box

#Valores para mandar el mensaje

# Access Slack credentials
token = st.secrets["slack"]["token"]
channel = st.secrets["slack"]["channel"]

# Access Foscam credentials
foscam_username = st.secrets["foscam"]["username"]
foscam_password = st.secrets["foscam"]["password"]
foscam_ip = st.secrets["foscam"]["ip"]
foscam_port = st.secrets["foscam"]["port"]
rtsp_url = f"rtsp://{foscam_username}:{foscam_password}@{foscam_ip}:{foscam_port}/videoMain"



def draw_boxes(frame, results, color=(255, 0, 0), label_prefix="", class_filter=None, 
               return_detected_objects=False, dibujar=True):
    """
    Dibuja bounding boxes y etiquetas en la imagen o devuelve objetos detectados.
    
    - `dibujar`: Si es False, no dibuja en el frame.
    - `class_filter`: Si se especifica, puede ser una lista de clases permitidas o una sola clase.
    - `return_detected_objects`: Si es True, devuelve también las cajas detectadas.
    """
    detected_objects = []  # Lista para almacenar las esquinas y clase de objetos detectados de interés
    
    for result in results:
        boxes = result.boxes  # Obtener las cajas delimitadoras
        for box in boxes:
            cls = int(box.cls[0])
            
            # Verificar si la clase detectada está en `class_filter` (soporta listas y enteros)
            if class_filter is not None and cls not in (class_filter if isinstance(class_filter, list) else [class_filter]):
                continue  # Salta las clases que no coinciden con `class_filter`

            cx, cy = box.xywh[0][0], box.xywh[0][1]  # Centro de la caja
            width, height = box.xywh[0][2], box.xywh[0][3]  # Dimensiones de la caja

            # Calcular coordenadas de las esquinas (x_min, y_min, x_max, y_max)
            x_min = int(cx - width / 2)
            y_min = int(cy - height / 2)
            x_max = int(cx + width / 2)
            y_max = int(cy + height / 2)

            # Extraer confianza
            conf = box.conf[0]

            # Asignar etiquetas personalizadas basadas en `label_prefix` y la clase
            if label_prefix == "Yolo1":
                # Para Yolo1, la etiqueta es "person" si la clase es 0
                label = "person" if cls == 0 else f"Class {cls}"
            elif label_prefix == "Yolo2":
                # Para Yolo2, la clase 0 es "helmet" y la clase 1 es "vest"
                if cls == 1:
                    label = "helmet"
                elif cls == 4:
                    label = "vest"
                else:
                    label = f"Class {cls}"
            else:
                label = f"Class {cls}"

            # Si dibujar es True, se dibuja el bounding box y el texto en la imagen
            if dibujar:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, f'{label_prefix}{label} {conf:.2f}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Si se debe devolver las cajas detectadas, almacenamos la información
            if return_detected_objects:
                detected_objects.append({
                    'class': cls,
                    'label': label,  # Se agrega la etiqueta personalizada
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                })

    # Si se requiere devolver las cajas detectadas, las devolvemos
    if return_detected_objects:
        return frame, detected_objects
    
    return frame, detected_objects
def verificar_proteccion(detected_objects: List[Dict[str, int]]) -> str:
    """
    Verifica si una persona está protegida basándose en el contacto de su bounding box con las cajas de clases específicas,
    como casco ("helmet") o chaleco ("vest").
    
    Parameters:
    - detected_objects: Lista de diccionarios con información sobre los objetos detectados.
    
    Returns:
    - str: Mensaje indicando si la persona está protegida o no.
    """
    # Etiquetas requeridas para protección
    etiquetas_proteccion = ["helmet", "vest"]
    persona_caja = None
    objetos_cajas = {etiqueta: [] for etiqueta in etiquetas_proteccion}

    # Convertir centros en cajas con esquinas (x_min, y_min, x_max, y_max)
    for obj in detected_objects:
        etiqueta = obj['label']
        x_min, y_min, x_max, y_max = obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']
        
        # Si la etiqueta es "person", asignamos la caja de la persona
        if etiqueta == "person":
            persona_caja = (x_min, y_min, x_max, y_max)
        elif etiqueta in etiquetas_proteccion:
            # Si es una etiqueta de protección, agregamos su caja
            objetos_cajas[etiqueta].append((x_min, y_min, x_max, y_max))

    # Si no se detecta una persona, retornar mensaje
    if persona_caja is None:
        return "No se detectó ninguna persona en la imagen.", None

    # Verificar si cada clase de protección está en contacto con la persona
    for etiqueta in etiquetas_proteccion:
        if objetos_cajas[etiqueta]:
            # Verificar si alguna caja de la etiqueta específica está en contacto con la persona
            en_contacto = any(cajas_en_contacto(persona_caja, caja_objeto) for caja_objeto in objetos_cajas[etiqueta])
            if not en_contacto:
                return f"El usuario no está protegido con el EPI adecuado: falta contacto con {etiqueta}.", 0
        else:
            return f"El usuario no está protegido con el EPI adecuado: falta {etiqueta}.", 0

    # Si todas las etiquetas requeridas están en contacto, el usuario está protegido
    return "El usuario está protegido con el EPI adecuado.", 1
def cajas_en_contacto(caja1: Tuple[int, int, int, int], caja2: Tuple[int, int, int, int]) -> bool:
    """
    Verifica si dos bounding boxes están en contacto.
    Cada bounding box se define como (x_min, y_min, x_max, y_max).
    """
    x1_min, y1_min, x1_max, y1_max = caja1
    x2_min, y2_min, x2_max, y2_max = caja2

    # Verificamos que no haya solapamiento en los ejes x y y
    if x1_max < x2_min or x2_max < x1_min:
        return False  # No hay solapamiento en x
    if y1_max < y2_min or y2_max < y1_min:
        return False  # No hay solapamiento en y

    # Si no hay espacios entre ellos, entonces están en contacto
    return True
def usuario_zona_peligrosa(x, y, sigma, detected_objects, frame):
    """
    Verifica si alguna persona (con etiqueta "person") está dentro de la zona peligrosa, 
    dibuja el área peligrosa en el frame y devuelve un mensaje de advertencia si está en esa zona.

    Parameters:
    - x, y: Coordenadas del centro de la zona peligrosa.
    - sigma: Mitad del lado del cuadrado que define la zona peligrosa.
    - detected_objects: Lista de objetos detectados con esquinas y etiquetas.
    - frame: El frame de video en el que se dibujará el área peligrosa.

    Returns:
    - str: Mensaje sobre si la persona está en la zona peligrosa o no.
    - int: 1 si está en la zona peligrosa, 0 si no lo está.
    """

    # Convertir la imagen a una matriz numpy en RGB o escala de grises
    image_array = np.array(frame)

    # Obtener la altura de la imagen
    height = image_array.shape[0]  # Esto te da la altura (número de filas)

    y = height - y

    # Definir las coordenadas de las esquinas de la zona peligrosa (un cuadrado con centro en (x, y) y lado 2*sigma)
    x_1 = int(x - sigma*0.5)
    y_1 = int(y + sigma*0.5)
    x_2 = int(x - sigma*0.5)
    y_2 = int(y - sigma*0.5)
    x_3 = int(x + sigma*0.5)
    y_3 = int(y - sigma*0.5)
    x_4 = int(x + sigma*0.5)
    y_4 = int(y + sigma*0.5)

    # Crear el polígono que representa el área peligrosa
    peligrosa_area = Polygon([(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)])

    # Dibujar el perímetro del área peligrosa en el frame en color blanco
    cv2.line(frame, (x_1, y_1), (x_2, y_2), (255, 255, 255), 2)
    cv2.line(frame, (x_2, y_2), (x_3, y_3), (255, 255, 255), 2)
    cv2.line(frame, (x_3, y_3), (x_4, y_4), (255, 255, 255), 2)
    cv2.line(frame, (x_4, y_4), (x_1, y_1), (255, 255, 255), 2)

    # Iterar sobre los objetos detectados para verificar si alguna persona está en la zona peligrosa
    for obj in detected_objects:
        # Verificar si la etiqueta es 'person'
        if obj['label'] == "person":
            # Calcular el punto central de la caja de la persona
            persona_box = box(obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max'])

            # Verificar si el punto central de la persona está dentro del área peligrosa
            if persona_box.contains(peligrosa_area):
                return "¡Usuario en zona peligrosa!", 1

    # Si no se encontró ninguna persona en la zona peligrosa
    return "Usuario fuera de la zona peligrosa", 0
def verificar_seguridad(epi, peligro):
    # Verificación de equipo de protección individual (epi)
    if epi == 0:
        mensaje_epi = "El usuario no lleva el equipo de seguridad adecuado."
    else:
        mensaje_epi = None
    
    # Verificación de nivel de peligro
    if peligro == 1:
        mensaje_peligro = "Peligro, trabajador localizado en zona de riesgo."
    else:
        mensaje_peligro = None
    
    # Juntar los mensajes si corresponde
    if mensaje_epi  and mensaje_peligro:
        return mensaje_epi + " " + mensaje_peligro
    elif mensaje_epi:
        return mensaje_epi
    elif mensaje_peligro:
        return mensaje_peligro
    return None  # Si no hay mensajes, devolver una cadena vacía
    #Mensaje a la plataforma 
    # Variable global para registrar el último tiempo de ejecución
ultima_ejecucion = 0

def send_slack_message(channel, message, token):
    global ultima_ejecucion
    
    # Tiempo actual
    ahora = time.time()
    
    # Verificar si han pasado al menos 5 segundos
    if ahora - ultima_ejecucion < 5:
        tiempo_restante = 5 - (ahora - ultima_ejecucion)
        print(f"Espera {tiempo_restante:.2f} segundos más para enviar otro mensaje.")
        return
    
    # Actualizar el tiempo de la última ejecución
    ultima_ejecucion = ahora

    # Verificar si el mensaje es None o está vacío
    if message is None or message.strip() == "":
        print("No se enviará ningún mensaje, ya que el contenido está vacío o es None.")
        return  # No enviar el mensaje si está vacío o es None

    # Enviar el mensaje a Slack
    client = WebClient(token=token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        print("Mensaje enviado con éxito!")
    except SlackApiError as e:
        print("Error al enviar mensaje:", e.response['error'])

def inference_dual_models(model1_path, model2_path):
    """Real-time object detection with two YOLO models in Streamlit."""
    st.set_page_config(page_title="Dual YOLO Streamlit App", layout="wide")

    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        model1 = YOLO(model1_path)
        model2 = YOLO(model2_path)

    st.markdown("<h1 style='text-align: center; color: #FF64DA;'>Aplicación para seguridad en construcción</h1>", 
                unsafe_allow_html=True)

    # UI para parámetros de zona peligrosa
    col1, col2 = st.columns([1, 1])
    with col2:
        st.write("Delimitación de zona peligrosa")
        x = st.number_input("X", min_value=0, value=0, key="val1")
        y = st.number_input("Y", min_value=0, value=0, key="val2")
        sigma = st.number_input("Longitud del lado del cuadrado", min_value=0, value=0, key="val3")

    # Cargar video
    uploaded_video = st.file_uploader("Sube un video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())

        cap = cv2.VideoCapture("temp_video.mp4")

        # Crear placeholders para los videos
        col1, col2 = st.columns(2)
        frame_placeholder1 = col1.empty()
        frame_placeholder2 = col2.empty()



        # Variable de control en session_state
        if "stop_processing" not in st.session_state:
            st.session_state.stop_processing = False

        # Botón para procesar el video
        if st.button("Procesar Video"):
            st.session_state.stop_processing = False  # Reset al iniciar

        # Botón para detener el video
        if st.button("Parar Video", key="stop_button"):
            st.session_state.stop_processing = True  # Cambia la variable de control

        # Cargar el video solo si el usuario no lo ha detenido
        if uploaded_video and not st.session_state.stop_processing:
            cap = cv2.VideoCapture("temp_video.mp4")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success or st.session_state.stop_processing:
                    st.warning("Fin del video o detenido por el usuario.")
                    break

                # Model 1 predictions
                results1 = model1.track(frame, conf=0.45, iou=0.45, classes=0, persist=True)

                # Model 2 predictions
                results2 = model2.track(frame, conf=0.45, iou=0.45, classes=[1, 4], persist=True)

                # Dibujar cajas
                frame, detected_objects1 = draw_boxes(frame, results1, color=(255, 0, 0), label_prefix="Yolo1", class_filter=0, return_detected_objects=True, dibujar=False)
                frame, detected_objects2 = draw_boxes(frame, results2, color=(0, 255, 0), label_prefix="Yolo2", class_filter=[1, 4], return_detected_objects=True, dibujar=False)

                # Anotaciones en los frames
                annotated_frame1 = results1[0].plot() if results1 else frame
                annotated_frame2 = results2[0].plot() if results2 else frame

                detected_objects = detected_objects1 + detected_objects2

                # Comprobaciones
                mensaje, epi = verificar_proteccion(detected_objects)
                mensaje, peligro = usuario_zona_peligrosa(x, y, sigma, detected_objects, annotated_frame1)

                # Mostrar frames en columnas separadas
                frame_placeholder1.image(annotated_frame1, channels="BGR", caption="Vigilancia de peligro")
                frame_placeholder2.image(annotated_frame2, channels="BGR", caption="Detección de EPI")

                # Enviar mensaje a Slack
                message = verificar_seguridad(epi, peligro)
                print(message)
                send_slack_message(channel, message, token)

                time.sleep(0.03)

        cap.release()
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()


inference_dual_models(model1_path="model_YOLO_1.pt", model2_path="model_YOLO_2.pt") 
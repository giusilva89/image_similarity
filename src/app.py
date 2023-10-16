import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
from skimage.metrics import structural_similarity as ssim
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2lab

# Set the path to your image directory
image_dir = "cats"

# Set the OMP_NUM_THREADS environment variable
os.environ['OMP_NUM_THREADS'] = '4'

def preprocess_images(image_files):
    images = []
    shapes = []
    colors = []
    for file in image_files:
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path, 1)
        resized_image = cv2.resize(image, (100, 100))
        images.append(resized_image)

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast').mean()
        correlation = greycoprops(glcm, 'correlation').mean()
        energy = greycoprops(glcm, 'energy').mean()
        homogeneity = greycoprops(glcm, 'homogeneity').mean()
        shape_features = [contrast, correlation, energy, homogeneity]
        shapes.append(shape_features)

        lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
        color_features = lab_image.mean(axis=(0, 1))
        colors.append(color_features)

    images = np.array(images)
    shapes = np.array(shapes)
    colors = np.array(colors)

    return images, shapes, colors

def cluster_evaluation(features, model):
    model = KMeans(n_init=10, random_state=42)
    visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion')
    visualizer.fit(features)
    #visualizer.show()
    optimal_clusters = visualizer.elbow_value_

    model = KMeans(n_clusters=optimal_clusters, random_state=42)
    model.fit(features)

    labels = model.labels_
    distortion = model.inertia_
    silhouette = silhouette_score(features, labels)
    calinski = calinski_harabasz_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)

    return labels, optimal_clusters, distortion, silhouette, calinski, davies_bouldin

def match_shape(input_image, image_path):
    input_image = cv2.resize(input_image, (100, 100))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    similarity = ssim(input_image, image)

    return similarity

def main():
    image_files = os.listdir(image_dir)
    images, shapes, colors = preprocess_images(image_files)

    shape_labels, optimal_shape_clusters, shape_distortion, shape_silhouette, shape_calinski, shape_davies_bouldin = cluster_evaluation(shapes, KMeans(random_state=42))

    color_labels, optimal_color_clusters, color_distortion, color_silhouette, color_calinski, color_davies_bouldin = cluster_evaluation(colors, KMeans(random_state=42))

    data = {'Image File': image_files, 'Shape Cluster Label': shape_labels, 'Color Cluster Label': color_labels}
    df = pd.DataFrame(data)
    df.to_csv("labels.csv", index=False)

    app = dash.Dash(__name__)

    styles = {
        'uploadButton': {
            'border': 'none',
            'background-color': '#F8F0E3',
            'color': '#0C090A',
            'padding': '10px 20px',
            'border-radius': '4px',
            'cursor': 'pointer',
            'margin': '20px auto',  # Center the button horizontally
            'font-family': 'Courier New',  # Set the font family to Calibri
            'width': '150px'  # Adjust the width of the button                
        },
        'imageContainer': {
            'display': 'flex',
            'flex-wrap': 'wrap',
            'justify-content': 'center',  # Center the images horizontally
            'align-items': 'center',  # Center the images vertically
            'text-align': 'center',
            'font-family': 'Courier New',  # Set the font family to Calibri        
        },
        'title': {
            'text-align': 'center',
            'margin': '10px 0',
            'font-size': '12px',  # Adjust the font size as desired
            'font-family': 'Courier New',  # Set the font family to Calibri
        },
        'image': {
            'width': '200px',  # Adjust the desired width for the output images
            'height': '200px',  # Adjust the desired height for the output images
            'object-fit': 'cover',  # Maintain aspect ratio and cover the available space
            'margin': '10px',
            'font-family': 'Courier New',  # Set the font family to Calibri
        },
        'spinnerContainer': {
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'height': '100vh',  # Set the height of the container to the full viewport height
            'background-color': 'white',  # Set the background color to white
            'font-family': 'Courier New',  # Set the font family to Calibri        
        }
    }
    
    app.layout = html.Div([
        html.H1("", style=styles['title']),
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div(
                    [
                        'Load Image',
                        html.I(className="fas fa-upload ml-2")
                    ],
                    className="custom-upload-button"
                ),
                style=styles['uploadButton']
            ),
        ], style={'text-align': 'center'}),
        html.Div([
            html.Div(id='input-image-container', style={'text-align': 'center'}),
            html.Div(id='output-images', style=styles['imageContainer']),
        ], style={'margin': '20px auto'}),
        html.Div(id='loading-spinner', children=[
            html.Div(className='spinner'),
            html.P('App is loading...')
        ], style=styles['spinnerContainer']),  # Loading spinner
    ])

    @app.callback(
        [Output('loading-spinner', 'style'), Output('input-image-container', 'children'), Output('output-images', 'children')],
        [Input('upload-image', 'contents')],
        [State('upload-image', 'filename')]
    )
    def update_app(contents, filename):
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded_image = cv2.imdecode(np.frombuffer(base64.b64decode(content_string), np.uint8), -1)
            decoded_image = cv2.resize(decoded_image, (100, 100))

            decoded_color_hist = cv2.calcHist([decoded_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            decoded_color_hist = cv2.normalize(decoded_color_hist, decoded_color_hist).flatten()

            color_hist_similarities = []
            shape_match_scores = []

            for i, image_filename in enumerate(df['Image File']):
                image_path = os.path.join(image_dir, image_filename)
                image = cv2.imread(image_path)
                image_color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                image_color_hist = cv2.normalize(image_color_hist, image_color_hist).flatten()

                color_hist_similarity = cv2.compareHist(decoded_color_hist, image_color_hist, cv2.HISTCMP_CORREL)
                color_hist_similarities.append(color_hist_similarity)

                shape_match_score = match_shape(decoded_image, image_path)
                shape_match_scores.append(shape_match_score)

            df['Color Hist Similarity'] = color_hist_similarities
            df['Shape Match Score'] = shape_match_scores

            df.sort_values(by=['Shape Match Score', 'Color Hist Similarity'], ascending=[False, True], inplace=True)

            input_image_html = html.Div([
                html.H3('Input Image', style=styles['title']),
                html.Img(src=contents, style=styles['image'])
            ])

            similar_images = []
            for i in range(5):
                image_filename = df.iloc[i]['Image File']
                image_path = os.path.join(image_dir, image_filename)
                image_data = cv2.imencode('.png', cv2.imread(image_path))[1].tostring()

                encoded_image = base64.b64encode(image_data).decode('utf-8')
                image = html.Div([
                    html.H3(f'Shape Match Score: {df.iloc[i]["Shape Match Score"]:.2f}', style=styles['title']),                    
                    html.H3(f'Color Hist Similarity: {df.iloc[i]["Color Hist Similarity"]:.2f}', style=styles['title']),
                    html.Img(src=f'data:image/png;base64,{encoded_image}', style=styles['image'])
                ])
                similar_images.append(image)

            return {'display': 'none'}, input_image_html, similar_images

        return {'display': 'flex'}, html.Div(), html.Div()

    if __name__ == '__main__':
        app.run_server(debug=False, port=8051)

if __name__ == "__main__":
    main()

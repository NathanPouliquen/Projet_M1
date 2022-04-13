#On importe les modules nécessaires
import imghdr
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Initialisation de la page web
st.title('Automated cross-section analysis')
st.write(" ------ ")

#Initialisation des constantes
IMAGE_DIR = 'images'
UPLOAD_DIR = 'upload'

# Constants for sidebar dropdown
SIDEBAR_OPTION_INTRO = "Introduction of the project"
SIDEBAR_OPTION_IMAGE = "Select an image"
SIDEBAR_OPTION_UPLOAD = "Upload an image"
SIDEBAR_OPTION_TEAM = "More about the team"

# Constants for actions
ACTION_OPTION_PHOTO = "Displaying the image"
ACTION_OPTION_CIRCLES = "Counting circles in the image"
ACTION_OPTION_GENERAL_COLOR = "Detection of predominant colours"
ACTION_OPTION_SPECIFIC_COLOR = "Select a specific colour"
ACTION_OPTION_CROPP = "Crop a part of the image"

# Constants for color
BLUE_OPTION = "Blue"
BROWN_OPTION = "Brown"
GREEN_OPTION = "Green"
ORANGE_OPTION = "Orange"
PURPLE_OPTION = "Purple"
RED_OPTION = "Red"
WHITE_OPTION = "White"
YELLOW_OPTION = "Yellow"

# Mise sous forme de liste pour les menus défillants
SIDEBAR_OPTIONS = [SIDEBAR_OPTION_INTRO, SIDEBAR_OPTION_IMAGE, SIDEBAR_OPTION_UPLOAD, SIDEBAR_OPTION_TEAM]
ACTION_OPTIONS = [ACTION_OPTION_PHOTO, ACTION_OPTION_CIRCLES, ACTION_OPTION_GENERAL_COLOR, ACTION_OPTION_SPECIFIC_COLOR, ACTION_OPTION_CROPP]
COLOR = [BLUE_OPTION, BROWN_OPTION, GREEN_OPTION, ORANGE_OPTION, PURPLE_OPTION, RED_OPTION, WHITE_OPTION, YELLOW_OPTION]

#Cette fonction va servir à afficher l'image dans l'app. On lui ajoute également un titre
def display_image_origin(img):
    st.image(img, caption = 'Image to be analysed')

#Cette fonction est utilisée pour la détection et le décompte des cercles dans l'image.
def decompte_ronds(img):

    #Chargement de l'image à analyser
    img = cv2.imread(img)

    #Positionnement des filtres appropriés    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurM = cv2.medianBlur(gray, 5)
    blurG = cv2.GaussianBlur(gray, (9, 9), 0)

    #Traitement de l'image en niveaux de gris avec cvtColor
    edge = cv2.Canny(gray, 100, 200) 
    edgeG = cv2.Canny(blurG, 100, 200) 
    edgeM = cv2.Canny(blurM, 100, 200)

    #Superposition et concaténation des différents filtres
    un=cv2.bitwise_or(edge,edgeG)
    deux=cv2.bitwise_or(un,edgeM)

    #Obtention de l'image traitée 
    gray = cv2.Canny(deux, 100, 200)
    
    #On va définir ici un slider qui va prendre un intervalle de valeur pour la taille des cercles à prendre en compte
    values = st.slider('On which radius values do you want to detect the circles ?',1, 14, (2, 12))
    lst_value=list(values)

    #Mise en place des différents seuils de détection des ronds
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows/80, param1=8, param2=8, minRadius=lst_value[0], maxRadius=lst_value[1])

    #Boucle permettant de repérer les cercles dans l'image et de les compter
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (255, 200, 100), 2)
            # circle outline(contour)
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 2)
    compteur=len(circles[0])

    #On imprime une image composée des différents cercles repérés
    cv2.imwrite("shape/detected_circles.jpg",img)
    image = Image.open('shape/detected_circles.jpg')
    st.image(image, caption="Circles detected on the image")

    #On fait un petit rapport sur les résultats obtenus
    st.write("The number of circles detected is :",compteur,".")
    st.write("The radius of the detected circles correspond to indicative values between", lst_value[0],"and", lst_value[1],".")

#Cette fonction va servir à repérer les couleurs dominantes en utilisant la méthode des K-Means.
def couleurs_predom(img):

    #Chargement de l'image à analyser
    img = cv2.imread(img)

    #On passe de BGR (par défaut en cv2) à RGB (le négatif)
    rgbs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_list = list(itertools.chain(*rgbs.tolist()))

    #On va définir ici un slider qui va définir le nombre de cluster que l'on va pouvoir former
    n_clusters = st.slider('How many predominant colours do you want to have ?', 1, 10, 5)
    #Méthode des k-means = Moyennage sur une infinité de couleurs possibles
    #Nombre de couleurs à extraire = Nombre de clusters (ex ici 5 par défaut):
    clusters = KMeans(n_clusters).fit(rgb_list)

    #Le centre du cluster est la couleur dominante, on obtient donc ici 5 couleurs majoritaires
    colors = clusters.cluster_centers_

    #On calcule ensuite le pourcentage de chaque cluster identifié
    def cluster_percents(labels):
        total = len(labels)
        percents = []
        for i in set(labels):
            percent = (np.count_nonzero(labels == i) / total) * 100
            percents.append(round(percent, 2))
        return percents

    #On va dessiner un graphique circulaire représentant les différentes proportions des couleurs
    #Echelle car la couleur de matplotlib n'accepte que les RVB mis à l'échelle pour des valeurs comprises entre 0 à 1.
    colors = clusters.cluster_centers_ / 255
    colors = colors.tolist()

    #On trie les couleurs de la plus grande proportion à la plus petite
    percents = cluster_percents(clusters.labels_)
    tup = zip(colors, percents)
    sorted_tup = sorted(tup, key=lambda n: n[1], reverse=True)
    sorted_colors = [c for c,p in sorted_tup]
    sorted_percents = [p for c,p in sorted_tup]

    #On crée le graphique
    fig, ax1 = plt.subplots()
    ax1.pie(sorted_percents, colors=sorted_colors, autopct='%1.1f%%', counterclock=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot((fig), transparent=True)

    #On fait un petit rapport sur les résultats obtenus
    st.write("The most represented colour highlighted by the K-Means method is the one with the largest proportion on the pie chart.")
    st.write("The percentage of this colour represents", sorted_percents[0], "percent of the total image analysed.")

#Cette fonction va permettre de sélectionner une couleur spécifique sur l'image et d'afficher uniquement celle-ci
def color(img, name, light, dark):

    #Chargement de l'image à analyser
    img = cv2.imread(img)

    #On passe de BGR (par défaut en cv2) à RGB (le négatif)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #On initialise notre masque que l'on va appliquer à notre image
    mask = cv2.inRange(img, dark, light)

    #La fonction analogue à un AND va superposer l'image à notre masque et voir s'il y a des match
    result = cv2.bitwise_and(img, img, mask=mask)

    #On imprime enfin notre image avec seulement les couleurs qui correspondent
    colour_save = 'colour/'+name+'.jpg'
    dark_square_save = 'colour/dark_square.jpg'
    light_square_save = 'colour/light_square.jpg'
    plt.imsave(colour_save,result)
    image = Image.open(colour_save)
    st.image(image, caption="Selected colour on the image")

    #Cette partie de la fonction sert à établir le pourcentage de remplissage de la couleur concernée

    #Échelle de gris
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #Nombre total de pixels
    whole_area = img_gray.size

    #Nombre de pixels dans la zone blanche
    white_area = cv2.countNonZero(img_gray)
    white_area = (white_area / whole_area * 100)
    white_area = round(white_area, 2)
    
    #On réalise un petit rapport sur les résultats obtenus 
    st.info("For information : Please note that the following calculations take into account the entire selected image. If you want to have a more precise analysis on a part of the image, please go to the image cropping section and select the area of interest.")
    st.write(" ------ ")
    st.write("The colours considered are from light :", light, "to dark :", dark,".")

    #Affichage de 2 carrés de couleurs pour montrer l'intervalle des couleurs prises en compte dans chaque cas
    l, d = st.columns(2)
    l_square = np.full((200, 200, 3), light, dtype=np.uint8) / 255.0
    d_square = np.full((200, 200, 3), dark, dtype=np.uint8) / 255.0
    plt.imsave(light_square_save,l_square)
    plt.imsave(dark_square_save,d_square)
    with l:
        st.subheader("Lightest colour :")
        image_l = Image.open(light_square_save)
        st.image(image_l)
    with d:
        st.subheader("Darkest colour :")
        image_d = Image.open(dark_square_save)
        st.image(image_d)
    st.write(" ------ ")
    st.write("Note that in this image, the fill rate of", name, "occupies", white_area, "percent of the image.")

#Cette dernière fonction sert à découper l'image afin de la recadrer et ne garder que les éléments d'intérêt
def crop(img):

    #Chargement de l'image à analyser
    img = cv2.imread(img)

    #On passe de BGR (par défaut en cv2) à RGB (le négatif)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # slider utilisé pour sélectionner les zones en fonction des abscisses et des ordonnées
    x = st.slider("x-scale", 0, img.shape[1], (0, img.shape[1]))
    y = st.slider("y-scale", 0, img.shape[0], (0, img.shape[0]))
    lst_x=list(x)
    lst_y=list(y)
    start = (lst_x[0], lst_y[0])
    end = (lst_x[1], lst_y[1])
    color = (0,127,255)

    # carré pour identifier la zone
    cv2.rectangle(img, start, end, color, 1)
    cropped_image = img[lst_y[0]:lst_y[1], lst_x[0]:lst_x[1]]
    st.image(cropped_image, caption ="New image")
    #Mise en place d'un boutton pour valider la procédure de découpage de l'image
    button=st.button("Validate")
    if button:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("images/croped_image.jpg", cropped_image)
        st.success("Your image has been successfully saved in the database, you can find it in the image selection tab and start an analysis.")

def main():

    #Partie de l'interface utilisateur qui va être affichée sur notre application
    st.sidebar.warning('In order to obtain SATISFACTORY RESULTS, please make sure that the image you want to analyse is of the BEST POSSIBLE QUALITY.')
    st.sidebar.title("Start the analysis :")

    app_mode = st.sidebar.selectbox("Please choose from the following options :", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_OPTION_INTRO:
        st.sidebar.write(" ------ ")
        st.sidebar.success("Project INFORMATION appears on the right !")
        st.header("Welcome to our app !")
        st.write("The automated analysis of thin sections of object samples (algae, plants, and others) for the quantification of physico-chemical parameters is an important issue in the field of microscopy observations.")
        explanation = st.expander("Why did we create this app ?")
        explanation.info("Currently, the majority of the analyses are performed manually by a human observer.")
        explanation.write("This poses several difficulties:")
        explanation.write("- The time needed to read and analyse the section can sometimes be very long and this will require the full attention of the technician.")
        explanation.write("- The evaluation of the sample is made very subjective by the laboratory technician (sight, sensitivity, precision, etc.) making the result very operator-dependent.")
        explanation.write("- The sequence of analyses can be very monotonous, and the technician's concentration cannot be the same for all the manipulations.")
        st.write(" ------ ")
        st.write("The aim of this application is to remove the subjective character of the analysis while facilitating the work of the manipulator. The analysis time is also reduced, and satisfactory results can be obtained with complete objectivity.")
        st.write("What are the main features of this app ?")
        feature1 = st.expander("Select an image from database or Upload it :")
        feature1.write("- The manipulator can choose an image from a pre-existing database, or he can drag and drop his own image which he wishes to analyse.") 
        feature1.info("These options can be selected on the sidebar on the left.")
        feature2 = st.expander("Counting circles in the image :")
        feature2.write("- A count of the number of circles as a function of radius can be performed with a report on the results obtained.")
        feature2.info("The following image is an example of what we can achieve with this feature.")
        feature2.image("information/image1.jpg")
        feature3 = st.expander("Detection of predominant colours :")
        feature3.write("- An analysis of the predominant colours can be performed.")
        feature3.info("The following image is an example of what we can achieve with this feature.")
        feature3.image("information/image2.jpg")
        feature4 = st.expander("Select a specific colour :")
        feature4.write("- A selection of a particular colour is possible to give an idea of the pigments present on the image with a report of the result of the analysis.")
        feature4.info("The following image is an example of what we can achieve with this feature.")
        feature4.image("information/image3.jpg")
        feature5 = st.expander("Crop a part of the image :")
        feature5.write("- The possibility to crop the image to focus the analysis on a particular element.")
        feature5.info("The following image is an example of what we can achieve with this feature.")
        feature5.image("information/image4.jpg")
    
    elif app_mode == SIDEBAR_OPTION_IMAGE:
        st.sidebar.write(" ------ ")

        directory = os.path.join(IMAGE_DIR)
        photos = []
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)
            # Find all valid images
            if imghdr.what(filepath) is not None:
                photos.append(file)
        photos.sort()

        option = st.sidebar.selectbox("Select an image registered in our database :", photos)
        st.sidebar.write(" ------ ")
        choice1 = os.path.join(directory, option)
        action = st.sidebar.selectbox("Select from the following actions :", ACTION_OPTIONS)
        if action == ACTION_OPTION_PHOTO:
            st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
            display_image_origin(choice1)   
        elif action == ACTION_OPTION_CIRCLES:
            st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
            display_image_origin(choice1)
            decompte_ronds(choice1)
        elif action == ACTION_OPTION_GENERAL_COLOR:
            st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
            display_image_origin(choice1)
            couleurs_predom(choice1)
        elif action == ACTION_OPTION_SPECIFIC_COLOR:
            st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
            display_image_origin(choice1)
            option2 = st.selectbox("Please select a colour :", COLOR)
            if option2 == BLUE_OPTION:
                color(choice1, 'blue', (120, 255, 255), (0, 105, 120))
            elif option2 == BROWN_OPTION:  
                color(choice1, 'brown', (130, 100, 40), (65, 40, 0))
            elif option2 == GREEN_OPTION:
                color(choice1, 'green', (90,200,95), (0, 75, 0))
            elif option2 == ORANGE_OPTION:  
                color(choice1, 'orange', (255, 115, 100), (130, 85, 0))
            elif option2 == PURPLE_OPTION:  
                color(choice1, 'purple', (255, 135, 255), (110, 0, 100))
            elif option2 == RED_OPTION:
                color(choice1, 'red', (255, 50, 100), (80, 0, 0))
            elif option2 == WHITE_OPTION:  
                color(choice1, 'white', (255, 255, 255), (145, 145, 145))
            elif option2 == YELLOW_OPTION:  
                color(choice1, 'yellow', (230, 220, 100), (150, 120, 0))
        elif action == ACTION_OPTION_CROPP:
            st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
            display_image_origin(choice1)
            crop(choice1)
        else :
            raise ValueError('Selected sidebar option is not implemented. Please open an issue on nathan.pouliquen@isen-ouest.yncrea.fr or emma.lucas@isen-ouest.yncrea.fr.')

    elif app_mode == SIDEBAR_OPTION_UPLOAD:
        st.sidebar.info("PRIVACY POLICY: Images will only be recorded with your consent. The use of this application is only intended to facilitate analysts in collecting data.")
        data = st.sidebar.file_uploader("Please select an image from your files that you wish to analyse :", type=['png', 'jpg', 'jpeg'])
        directory = os.path.join(UPLOAD_DIR)
        if data is not None:
            for files in os.listdir(directory):
                os.remove(os.path.join(directory, files))
            with open(os.path.join(directory,data.name),"wb") as f:
                f.write((data).getbuffer())
            for file in os.listdir(directory):
                choice2 = os.path.join(directory, file)
            action = st.sidebar.selectbox("Select from the following actions :", ACTION_OPTIONS)
            if action == ACTION_OPTION_PHOTO:
                st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
                display_image_origin(choice2)
            elif action == ACTION_OPTION_CIRCLES:
                st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
                display_image_origin(choice2)
                decompte_ronds(choice2)
            elif action == ACTION_OPTION_GENERAL_COLOR:
                st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
                display_image_origin(choice2)
                couleurs_predom(choice2)
            elif action == ACTION_OPTION_SPECIFIC_COLOR:
                st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
                display_image_origin(choice2)
                option2 = st.selectbox("Please select a colour :", COLOR)
                if option2 == BLUE_OPTION:
                    color(choice2, 'blue', (120, 255, 255), (0, 105, 120))
                elif option2 == BROWN_OPTION:  
                    color(choice2, 'brown', (130, 100, 40), (65, 40, 0))
                elif option2 == GREEN_OPTION:
                    color(choice2, 'green', (90,200,95), (0, 75, 0))
                elif option2 == ORANGE_OPTION:  
                    color(choice2, 'orange', (255, 115, 100), (130, 85, 0))
                elif option2 == PURPLE_OPTION:  
                    color(choice2, 'purple', (255, 135, 255), (110, 0, 100))
                elif option2 == RED_OPTION:
                    color(choice2, 'red', (255, 50, 100), (80, 0, 0))
                elif option2 == WHITE_OPTION:  
                    color(choice2, 'white', (255, 255, 255), (145, 145, 145))
                elif option2 == YELLOW_OPTION:  
                    color(choice2, 'yellow', (230, 220, 100), (150, 120, 0))
            elif action == ACTION_OPTION_CROPP:
                st.sidebar.success("WAIT a moment, the analysis will be displayed on the RIGHT SIDE.")
                display_image_origin(choice2)
                crop(choice2)
            else :
                raise ValueError('Selected sidebar option is not implemented. Please open an issue on nathan.pouliquen@isen-ouest.yncrea.fr or emma.lucas@isen-ouest.yncrea.fr.')

    elif app_mode == SIDEBAR_OPTION_TEAM:
        st.sidebar.write(" ------ ")
        st.subheader("Presentation of the team :")
        st.image("information/ISEN.jpg", width = 300)
        st.write("We are students in the second year of the engineering cycle at ISEN in Brest. We carried out this project as part of our studies to validate the skills acquired throughout the year and also to facilitate the work of laboratory technicians by reducing analysis time and making the data collected as reliable as possible.")
        st.sidebar.write('Please feel free to connect with us on Linkedin !')
        st.sidebar.write(" ------ ")
        st.sidebar.success('We hope you enjoyed the features available !')
        expandar_linkedin = st.expander('Contact Information')
        expandar_linkedin.write('Emma LUCAS: https://www.linkedin.com/in/emma-lucas-972049173/')
        expandar_linkedin.write('Nathan POULIQUEN: https://www.linkedin.com/in/nathan-pouliquen/')
    else:
        raise ValueError('Selected sidebar option is not implemented. Please open an issue on nathan.pouliquen@isen-ouest.yncrea.fr or emma.lucas@isen-ouest.yncrea.fr.')

main()

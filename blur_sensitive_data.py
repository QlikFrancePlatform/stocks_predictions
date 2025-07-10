#!/usr/bin/env python3
"""
Script pour flouter automatiquement les informations sensibles dans les images
"""

from PIL import Image, ImageFilter, ImageDraw
import re
import os

def blur_sensitive_areas(image_path, output_path=None):
    """
    Floute les zones sensibles dans une image (tokens, mots de passe, etc.)
    """
    if output_path is None:
        # Créer un nom de fichier avec "_blurred" ajouté
        name, ext = os.path.splitext(image_path)
        output_path = f"{name}_blurred{ext}"
    
    # Ouvrir l'image
    img = Image.open(image_path)
    
    # Convertir en RGB si nécessaire
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Créer une copie pour le floutage
    blurred_img = img.copy()
    
    # Appliquer un flou gaussien à toute l'image
    blurred_img = blurred_img.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Créer un masque pour les zones sensibles
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Définir les zones sensibles (à ajuster selon votre image)
    # Ces coordonnées sont approximatives, à ajuster selon votre image
    sensitive_areas = [
        # Zone pour les tokens (à ajuster selon votre image)
        (200, 150, 400, 200),  # Exemple de zone
        (300, 250, 500, 300),  # Autre zone possible
        # Ajoutez d'autres zones selon votre image
    ]
    
    # Dessiner les zones sensibles en blanc sur le masque
    for area in sensitive_areas:
        draw.rectangle(area, fill=255)
    
    # Combiner l'image originale avec l'image floutée
    result = Image.composite(img, blurred_img, mask)
    
    # Sauvegarder l'image floutée
    result.save(output_path)
    print(f"Image floutée sauvegardée : {output_path}")
    
    return output_path

def blur_all_assets():
    """
    Floute toutes les images dans le dossier assets
    """
    assets_dir = "assets"
    
    if not os.path.exists(assets_dir):
        print(f"Dossier {assets_dir} non trouvé")
        return
    
    for filename in os.listdir(assets_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(assets_dir, filename)
            print(f"Traitement de {filename}...")
            blur_sensitive_areas(image_path)

if __name__ == "__main__":
    print("🔒 Floutage des informations sensibles dans les images...")
    
    # Option 1 : Flouter une image spécifique
    # blur_sensitive_areas("assets/connector2.png")
    
    # Option 2 : Flouter toutes les images du dossier assets
    blur_all_assets()
    
    print("✅ Floutage terminé !") 
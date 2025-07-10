#!/usr/bin/env python3
"""
Script simple pour flouter les tokens sensibles dans connector2.png
"""

from PIL import Image, ImageFilter, ImageDraw
import os

def blur_token_areas():
    """
    Floute les zones contenant des tokens dans connector2.png
    """
    input_path = "assets/connector2.png"
    output_path = "assets/connector2_blurred.png"
    
    if not os.path.exists(input_path):
        print(f"❌ Fichier {input_path} non trouvé")
        return
    
    print(f"🔒 Floutage des tokens dans {input_path}...")
    
    # Ouvrir l'image
    img = Image.open(input_path)
    
    # Convertir en RGB si nécessaire
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Créer une copie pour le floutage
    blurred_img = img.copy()
    
    # Appliquer un flou gaussien
    blurred_img = blurred_img.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Créer un masque pour les zones sensibles
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Définir les zones sensibles (à ajuster selon votre image)
    # Ces coordonnées sont approximatives - vous devrez les ajuster
    sensitive_areas = [
        # Zone pour les bearer tokens (exemple)
        (100, 200, 400, 250),  # Ajustez selon votre image
        (150, 300, 450, 350),  # Autre zone possible
        (200, 400, 500, 450),  # Troisième zone
        # Ajoutez d'autres zones selon votre image
    ]
    
    # Dessiner les zones sensibles en blanc sur le masque
    for area in sensitive_areas:
        draw.rectangle(area, fill=255)
    
    # Combiner l'image originale avec l'image floutée
    result = Image.composite(img, blurred_img, mask)
    
    # Sauvegarder l'image floutée
    result.save(output_path)
    print(f"✅ Image floutée sauvegardée : {output_path}")
    
    return output_path

if __name__ == "__main__":
    blur_token_areas() 
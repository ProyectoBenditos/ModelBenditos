import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ── CONFIGURACIÓN ─────────────────────────────────────
MODELO_PATH = 'best_model.pth'
IMSIZE      = 64
device      = torch.device('cpu')  # CPU en local, sin GPU necesaria

# ── CARGAR MODELO ─────────────────────────────────────
def cargar_modelo():
    """Carga el modelo ResNet18 pre-entrenado y fine-tuned para detección de cáncer pulmonar"""
    try:
        model = models.resnet18(weights=None)
        
        # Congelar capas tempranas (transfer learning)
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
                
        # Reemplazar capa fully connected para clasificación binaria
        model.fc = nn.Sequential(
            nn.Dropout(0.5),        # Regularización para evitar overfitting
            nn.Linear(512, 64),     # Capa densa intermedia
            nn.ReLU(),               # Activación no lineal
            nn.Linear(64, 1),       # Salida binaria
            nn.Sigmoid()             # Probabilidad entre 0 y 1
        )
        
        # Cargar pesos entrenados
        model.load_state_dict(torch.load(MODELO_PATH, map_location=device))
        model.eval()  # Modo evaluación (desactiva dropout y batch norm)
        return model
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra el archivo '{MODELO_PATH}'")
        print("   Asegúrate de que el modelo entrenado esté en la misma carpeta.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)

# ── NIVEL DE RIESGO ───────────────────────────────────
def nivel_riesgo(score):
    """Clasifica el score en nivel de riesgo y color asociado"""
    if score < 0.33:   
        return "BAJO",  "#2ecc71"  # Verde
    elif score < 0.66: 
        return "MEDIO", "#f39c12"  # Naranja
    else:              
        return "ALTO",  "#e74c3c"  # Rojo

# ── OBTENER RECOMENDACIÓN MÉDICA ───────────────────────
def obtener_recomendacion(nivel, score):
    """Genera recomendación médica basada en el nivel de riesgo"""
    if nivel == "ALTO":
        return "Evaluación clínica urgente y posible biopsia"
    elif nivel == "MEDIO":
        return "Seguimiento con CT en 3-6 meses"
    else:
        return "Control rutinario anual recomendado"

# ── ANALIZAR IMAGEN ───────────────────────────────────
def analizar(model, img_path):
    """Procesa una imagen CT y genera visualización de resultados"""
    try:
        # Cargar imagen original y preprocesar
        img_orig = np.array(Image.open(img_path).convert('L'))
        img      = Image.open(img_path).convert('L').resize((IMSIZE, IMSIZE))
        img_arr  = np.array(img, dtype=np.float32) / 255.0
        
        # Convertir a tensor 3 canales (ResNet espera RGB)
        tensor   = torch.tensor(
            np.stack([img_arr, img_arr, img_arr], axis=0)
        ).unsqueeze(0)

        # Inferencia
        with torch.no_grad():
            score = model(tensor).item()

        nivel, color = nivel_riesgo(score)
        recomendacion = obtener_recomendacion(nivel, score)

        # ── VISUALIZACIÓN ─────────────────────────────────
        fig = plt.figure(figsize=(14, 5))
        fig.suptitle('OncaScan Platform — Análisis de CT Pulmonar',
                     fontsize=15, fontweight='bold')

        # 1. Imagen CT original
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img_orig, cmap='gray')
        ax1.set_title('CT Pulmonar Analizada', fontsize=12)
        ax1.axis('off')

        # 2. Panel de resultados
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Fondo coloreado según riesgo
        rect = plt.Rectangle((0.05, 0.45), 0.9, 0.45,
                             color=color, alpha=0.15, linewidth=2, 
                             edgecolor=color, transform=ax2.transAxes)
        ax2.add_patch(rect)
        
        # Textos informativos
        ax2.text(0.5, 0.85, f'RIESGO {nivel}',
                 ha='center', fontsize=22, fontweight='bold', 
                 color=color, transform=ax2.transAxes)
        ax2.text(0.5, 0.70, f'Score: {score:.4f}',
                 ha='center', fontsize=14, color=color, 
                 transform=ax2.transAxes)
        
        # Detección de nódulo (threshold 0.5)
        deteccion = '✓ Nódulo detectado' if score >= 0.5 else '○ Sin nódulo detectado'
        ax2.text(0.5, 0.55, deteccion,
                 ha='center', fontsize=13, color='#555555', 
                 transform=ax2.transAxes)
        
        # Recomendación médica
        ax2.text(0.5, 0.25, recomendacion,
                 ha='center', fontsize=10, color='black',
                 style='italic', transform=ax2.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                          edgecolor='gray', alpha=0.7))
        
        ax2.set_title('Resultado del Modelo', fontsize=12)

        # 3. Medidor de riesgo visual
        ax3 = fig.add_subplot(1, 3, 3)
        
        # Barras de colores para los niveles
        for ancho, c, left in zip(
            [0.33, 0.33, 0.34],
            ['#2ecc71', '#f39c12', '#e74c3c'],
            [0, 0.33, 0.66]
        ):
            ax3.barh(0, ancho, left=left, color=c, height=0.4, alpha=0.7)
        
        # Indicador de score actual
        ax3.barh(0, 0.015, left=max(0, min(score-0.007, 0.985)),
                 color='black', height=0.65)
        
        # Configuración del gráfico
        ax3.set_xlim(0, 1)
        ax3.set_ylim(-0.6, 0.9)
        ax3.set_yticks([])
        ax3.set_xlabel('Score de Riesgo', fontsize=11)
        
        # Etiquetas de niveles
        ax3.text(0.165, 0.38, 'BAJO',  ha='center', fontsize=10,
                 fontweight='bold', color='#27ae60')
        ax3.text(0.495, 0.38, 'MEDIO', ha='center', fontsize=10,
                 fontweight='bold', color='#e67e22')
        ax3.text(0.825, 0.38, 'ALTO',  ha='center', fontsize=10,
                 fontweight='bold', color='#c0392b')
        
        # Valor numérico del score
        ax3.text(score, -0.45, f'{score:.3f}',
                 ha='center', fontsize=12, fontweight='bold')
        
        ax3.set_title('Nivel de Riesgo', fontsize=12)

        plt.tight_layout()
        
        # Guardar resultado
        nombre_base = os.path.splitext(os.path.basename(img_path))[0]
        nombre_salida = f'resultado_{nombre_base}.png'
        plt.savefig(nombre_salida, dpi=150, bbox_inches='tight')
        
        # Mostrar en consola
        print(f"\n{'='*50}")
        print(f"📊 RESULTADO DEL ANÁLISIS")
        print(f"{'='*50}")
        print(f"  📁 Archivo:    {img_path}")
        print(f"  📈 Score:      {score:.4f}")
        print(f"  ⚠️  Riesgo:     {nivel}")
        print(f"  💡 Recomendación: {recomendacion}")
        print(f"  💾 Guardado como: {nombre_salida}")
        print(f"{'='*50}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error al analizar {img_path}: {e}")

# ── SELECCIONAR MÚLTIPLES IMÁGENES ─────────────────────
def seleccionar_imagenes():
    """Abre un diálogo para seleccionar múltiples imágenes"""
    try:
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana principal de tkinter
        root.attributes('-topmost', True)  # Ventana siempre al frente
        
        archivos = filedialog.askopenfilenames(
            title="Seleccionar imágenes CT pulmonar para analizar",
            filetypes=[
                ("Todas las imágenes", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("TIFF", "*.tiff"),
                ("Todos los archivos", "*.*")
            ],
            initialdir=os.getcwd()  # Empezar en directorio actual
        )
        
        root.destroy()
        return list(archivos)
    except Exception as e:
        print(f"❌ Error al abrir selector de archivos: {e}")
        print("   Asegúrate de tener tkinter instalado correctamente.")
        return []

# ── MAIN ──────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🔬 ONCASCAN PLATFORM - Detección de Cáncer Pulmonar")
    print("="*50)
    
    # Cargar modelo
    print("\n⏳ Cargando modelo de IA...")
    model = cargar_modelo()
    print("✅ Modelo cargado correctamente\n")

    # Modo de operación
    if len(sys.argv) > 1:
        # Modo línea de comandos: python demo.py imagen.png
        print("📂 Modo: Línea de comandos")
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"❌ Error: No se encuentra el archivo '{img_path}'")
            sys.exit(1)
        analizar(model, img_path)
        
    else:
        # Modo selector interactivo
        print("📂 Modo: Selector interactivo")
        print("   Se abrirá una ventana para seleccionar imágenes...")
        print("   💡 Tip: Puedes seleccionar múltiples archivos con Ctrl+Click\n")
        
        imagenes = seleccionar_imagenes()
        
        if not imagenes:
            print("\n❌ No se seleccionó ninguna imagen.")
            print("\n📖 Opciones de uso:")
            print("   1. Ejecutar sin argumentos para abrir selector gráfico")
            print("   2. python demo.py imagen.png (analiza una imagen específica)")
            print("   3. Coloca imágenes en la carpeta y selecciónalas con el diálogo\n")
        else:
            print(f"\n✅ {len(imagenes)} imagen(es) seleccionada(s) para análisis")
            print("   Iniciando procesamiento...\n")
            
            for i, img in enumerate(imagenes, 1):
                print(f"\n[{i}/{len(imagenes)}] Analizando: {os.path.basename(img)}")
                analizar(model, img)
            
            print("\n" + "="*50)
            print("✅ ANÁLISIS COMPLETADO")
            print("="*50)
            print(f"   Se procesaron {len(imagenes)} imágenes")
            print("   Los resultados se guardaron como 'resultado_*.png'")
            print("="*50)
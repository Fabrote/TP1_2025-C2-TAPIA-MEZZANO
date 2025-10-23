# TP1 - Procesamiento De Imágenes

**Autores:** Tapia Fabrizio - Mezzano Florencia  
**Universidad:** FCEIA - TUIA

---

## Descripción

Este trabajo práctico implementa dos ejercicios de procesamiento de imágenes:

1. **Ecualización Local de Histograma** - Mejora de contraste en imágenes con detalles ocultos
2. **Validación Automática de Formularios** - Sistema de validación de formularios mediante análisis de componentes conectadas

---

## Estructura del Proyecto

```
TP1/
├── E1.PY                                    # Ejercicio 1: Ecualización local
├── E2.PY                                    # Ejercicio 2: Validación de formularios
├── Imagen_con_detalles_escondidos.tif      # Imagen de entrada para E1
├── formulario_01.png                        # Formularios de prueba (tipo A)
├── formulario_02.png                        # 
├── formulario_03.png                        # 
├── formulario_04.png                        #
├── formulario_05.png                        # 
├── formulario_vacio.png                     # Plantilla vacía
├── resultado_formularios.png                # Salida visual de validación
├── resultados_validacion.csv                # Resultados en formato CSV
├── comparacion_ventanas.png                 # Comparación de tamaños de ventana (E1)
├── imagen_ecualizada_local.png             # Resultado de ecualización (E1)
└── README.md                                # Este archivo
```

---

## Requisitos

### Dependencias

```bash
pip install opencv-python numpy matplotlib pandas
```

### Versiones recomendadas
- Python 3.8+
- OpenCV 4.x
- NumPy 1.20+
- Matplotlib 3.x
- Pandas 1.x

---

## Uso

### Ejercicio 1: Ecualización Local de Histograma

```bash
python E1.PY
```

**Funcionalidad:**
- Aplica ecualización local de histograma con ventanas de diferentes tamaños (11x11, 21x21, 41x41, 51x51, 101x101)
- Compara resultados visualmente con histogramas
- Revela detalles ocultos en imágenes de bajo contraste

**Parámetros configurables:**
- `window_size`: Tamaño de la ventana de ecualización (debe ser impar)

**Salida:**
- Visualización comparativa de diferentes tamaños de ventana
- Histogramas antes y después de la ecualización

---

### Ejercicio 2: Validación de Formularios

```bash
python E2.PY
```

**Funcionalidad:**
- Detecta automáticamente el tipo de formulario (A, B o C)
- Extrae y valida campos mediante análisis de componentes conectadas
- Genera reporte visual y CSV con resultados

**Campos validados:**
1. **Nombre y Apellido**: Mínimo 2 palabras, máximo 25 caracteres
2. **Edad**: 2-3 dígitos consecutivos
3. **Mail**: 1 palabra, máximo 25 caracteres
4. **Legajo**: Exactamente 8 caracteres consecutivos
5. **Preguntas (3)**: Una única opción marcada (Sí/No)
6. **Comentarios**: Al menos 1 palabra, máximo 25 caracteres

**Salidas:**
- `resultado_formularios.png`: Imagen con indicadores visuales (verde=OK, rojo=MAL)
- `resultados_validacion.csv`: Tabla con resultados de validación
- Reporte en consola agrupado por tipo de formulario

---


## Resultados

### Ejercicio 1
- **Entrada**: Imagen TIF con detalles ocultos
- **Salida**: Comparación visual de 5 tamaños de ventana
- **Mejor resultado**: Ventana 51x51 (balance entre detalle y ruido)

### Ejercicio 2
- **Formularios procesados**: 5
- **Tasa de detección**: 100%
- **Campos validados por formulario**: 8
- **Formato de salida**: PNG + CSV

---



## Archivos de Salida

| Archivo | Descripción |
|---------|-------------|
| `comparacion_ventanas.png` | Comparación visual de tamaños de ventana (E1) |
| `imagen_ecualizada_local.png` | Resultado final de ecualización (E1) |
| `resultado_formularios.png` | Validación visual con indicadores de color (E2) |
| `resultados_validacion.csv` | Tabla de resultados por campo (E2) |

---



## Autores

- **Tapia Fabrizio**
- **Mezzano Florencia**

**Institución:** TUIA  
**Materia:** Procesamiento de imágenes
**Año:** 2025 - Cuatrimestre 2

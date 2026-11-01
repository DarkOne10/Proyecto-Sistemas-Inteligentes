# Agente RL para esquivar ataques de Toriel

Simulación 2D propia inspirada en el combate de Toriel. Incluye Paw Sweep,
Fire Waves, Fire Helix y Faltering; reglas de HP y Spare; agente Q-Learning
tabular; entrenamiento, evaluación y demo con Pygame. No automatiza ni
modifica Undertale.

La guía completa de conceptos, instalación, comandos y resultados está en
[GUIA_EJECUCION.md](GUIA_EJECUCION.md).

## Requisitos e instalación

Se requiere Python 3.10 o posterior y conexión a internet para instalar las
dependencias. Desde la carpeta del proyecto:

```powershell
py -m venv .venv
 .\.venv\Scripts\python.exe -m pip install -e ".[render,test]"
```

En Windows no es necesario activar el entorno virtual. En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[render,test]"
```

Los comandos de ejecución para Windows están en [GUIA_EJECUCION.md](GUIA_EJECUCION.md).

## Ejecutar

Abrir la simulación con una política aleatoria:

```bash
python -m toriel_rl.main --seed 7
```

Entrenar Q-Learning con 2,000 episodios de práctica aislada (rotando los tres
ataques dañinos), seguidos de 500 combates completos de fine-tuning. El modelo
y los registros se guardan en `results/`:

```bash
python -m toriel_rl.training --episodes 2000 --full-combat-episodes 500 --seed 7
```

Evaluar cada patrón y combates completos contra una política aleatoria usando
las mismas semillas:

```bash
python -m toriel_rl.evaluation --model results/q_table.json --episodes 100
```

Ver la política aprendida en la ventana:

```bash
python -m toriel_rl.main --model results/q_table.json --seed 7
```

Para probar un solo patrón con el modelo, sin jugar las 12 rondas:

```bash
python -m toriel_rl.main --model results/q_table.json --attack fire_waves --attack-only
```

El modelo guardado usa el formato de observación vigente; vuelve a entrenar si
el programa avisa que una tabla Q antigua es incompatible.

## Pruebas

```bash
python -m unittest discover -s tests
```

Los patrones y colisiones son aproximaciones para el proyecto, no una réplica
visual o física exacta del juego. Paw Sweep puede generar una segunda pata en
rondas posteriores; Fire Waves selecciona ocasionalmente su variante difícil.

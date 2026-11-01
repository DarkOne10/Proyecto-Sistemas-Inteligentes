# Proyecto-Sistemas-Inteligentes

## Entorno Toriel RL

Prototipo de una simulación 2D para entrenar agentes que esquivan patrones
inspirados en la batalla contra Toriel. El entorno inicial implementa Fire
Helix, variación reproducible por episodio y la transición a Faltering al
quedar con 2 HP o menos.

### Instalacion

```bash
python -m pip install -e ".[render,test]"
```

### Demo

```bash
python -m toriel_rl.main --seed 7
```

### Pruebas

```bash
python -m unittest discover -s tests
```

`TorielEnv` sigue la interfaz de Gymnasium. Las semillas permiten reproducir
escenarios; cada episodio conserva sus reglas y parámetros.
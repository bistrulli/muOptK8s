# HPA Stabilization Window Simulation - Modular Structure

Questo progetto simula il comportamento di un sistema di autoscaling HPA (Horizontal Pod Autoscaler) di Kubernetes, con particolare attenzione all'impatto delle finestre di stabilizzazione e dei ritardi di attuazione.

## Struttura Modulare

Il progetto è stato organizzato in moduli separati per migliorare la manutenibilità e la riusabilità del codice:

### Moduli Principali

- **`application_server.py`** - Classe `ApplicationServer`
  - Rappresenta il centro di servizio multi-server (G/G/c) con disciplina FIFO
  - Gestisce il processamento delle richieste, metriche CPU e scaling delle repliche
  - Include logica di warm-up per le nuove repliche

- **`workload_generator.py`** - Classe `WorkloadGenerator`
  - Genera richieste seguendo un processo di Poisson
  - Permette di modificare dinamicamente il tasso di arrivo delle richieste
  - Simula carichi di lavoro fluttuanti

- **`hpa_controller.py`** - Classe `HPAController`
  - Implementa la logica del controller HPA con finestre di stabilizzazione
  - Gestisce le decisioni di scaling basate sull'utilizzo CPU
  - Include ritardi di attuazione e logica di stabilizzazione

- **`simulation_utils.py`** - Funzioni di utilità
  - `create_workload_scenario()` - Crea scenari di carico fluttuante
  - `data_logger()` - Registra metriche del sistema
  - `plot_results()` - Visualizza risultati di singole simulazioni
  - `plot_stabilization_comparison()` - Confronta diverse finestre di stabilizzazione

- **`hpa_simulation.py`** - Script principale
  - Orchestrazione delle simulazioni
  - Confronto tra diverse configurazioni di finestre di stabilizzazione
  - Analisi dei risultati

### File di Supporto

- **`validate_application.py`** - Script di validazione del modello ApplicationServer
- **`requirements.txt`** - Dipendenze Python necessarie

## Installazione e Utilizzo

1. **Attivare l'ambiente virtuale:**
   ```bash
   source .venv/bin/activate
   ```

2. **Installare le dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Eseguire la simulazione principale:**
   ```bash
   python hpa_simulation.py
   ```

4. **Validare il modello:**
   ```bash
   python validate_application.py
   ```

## Parametri di Simulazione

- **Durata simulazione:** 800 secondi
- **Repliche iniziali:** 1
- **CPU per replica:** 1
- **Target CPU utilization:** 70%
- **Intervallo di valutazione:** 15 secondi
- **Ritardo di attuazione:** 60 secondi
- **Tempo di warm-up:** 30 secondi

## Finestre di Stabilizzazione Testate

- **Upscale window:** 0 secondi (fisso)
- **Downscale window:** 0, 60, 180, 300 secondi

## Risultati

La simulazione confronta l'impatto di diverse finestre di stabilizzazione downscale su:
- Throughput durante periodi di alto carico
- Tempo di risposta medio
- Numero di azioni di scaling eseguite vs bloccate
- Timeline delle repliche

## Vantaggi della Struttura Modulare

1. **Separazione delle responsabilità:** Ogni classe ha un ruolo specifico e ben definito
2. **Riusabilità:** I moduli possono essere facilmente riutilizzati in altri progetti
3. **Manutenibilità:** Modifiche a un componente non influenzano gli altri
4. **Testabilità:** Ogni modulo può essere testato indipendentemente
5. **Leggibilità:** Il codice è più organizzato e facile da comprendere

## Estensioni Future

La struttura modulare facilita l'aggiunta di nuove funzionalità:
- Nuovi tipi di controller (VPA, KEDA, etc.)
- Diversi modelli di workload
- Metriche aggiuntive
- Algoritmi di scaling personalizzati

# Change Detection con Active Learning

Repository per il progetto di tesi "Sintesi di un algoritmo di Active Learning per Change Detection in immagini multispettrali bi-temporali", A.A. 2020/2021.

Il dataset utilizzato è reperibile al seguente link: https://cutt.ly/NbDeLgT
## Struttura del repository

    
        .
        ├── Net       	                # contiene i vari file con i dati necessari alla computazione (file .txt, .csv, .py)
        │    ├ data         		# locazione dei dataset da utilizzare
        │    │  ├ oneratest                 # locazione del dataset Onera (test)
        │    │  │     ├ cluster             # contiene i dizionari dei cluster serializzati per le coppie del dataset Onera (test)
        │    │  │     └ pseudo              # contiene le pseudo-etichette selezionate per le coppie del dataset Onera (test)
        │    │  └ oneratrain                # locazione del dataset Onera (train)
        │    │        └ pseudo              # contiene le pseudo-etichette selezionate per le coppie del dataset Onera (train)
        │    ├ model                        # locazione per il salvataggio e il caricamento dei modelli appresi
        │    │  └ model.old 		# contiene tutti i modelli ricavati
        │    ├ stat           		# locazione per il salvataggio delle metriche e statistiche ricavate durante la sperimentazione
        │    │  └ stat.old                  # contiene tutti i file .csv e le mappe .png dei modelli
        │    ├ config.py                    # file contenente le costanti "interne" per l'esecuzione degli script
        │    ├ contenuti.txt                # file di testo che dettaglia i contenuti delle cartelle stat, model e data     
        │    ├ dataprocessing.py            # file contenente le funzioni di caricamento e processing del dataset
        │    ├ labelbyneigh_generation.py   # script per la generazione delle mappe di pseudolabel per vicinato
        │    ├ labelbyperc_generation.py    # script per la generazione delle mappe di pseudolabel per percentuale
        │    ├ main.py                      # file contenente lo script principale per il programma
        │    ├ net.conf                     # file di configurazione per i vari dataset e per gli algoritmi implementati
        │    ├ predutils.py                 # file contenente funzioni di supporto alla predizione
        │    ├ Report riassuntivo.xlsx      # file contenente i risultati di tutti i test eseguiti
        │    ├ requirements.txt             # file contenente la lista delle dependencies necessarie
        │    └ siamese.py                   # file contenente le funzioni per il training e il fine tuning delle reti
        └── README.md

Sia i modelli che i file contenenti metriche e statistiche in model.old e stat.old sono suddivisi in un sistema di sottocartelle, ciascuna indicante un tipo di esperimento.

### Modelli
Ciascun modello appreso viene salvato in due tipi di file:

    nomemodello.h5              # file contenente i pesi appresi durante la fase di training
    nomemodello_param.pickle    # file contenente un dizionario serializzato con le informazioni utili 
                                  alla costruzione dell'architettura di rete

Generalmente il nome dei modelli è stato ottenuto associando le iniziali del dataset ("BA", "SB", "OTR", "OTE") alla sigla della distanza ("ED" o "SAM")
più ulteriori sottostringhe per indicarne alcune caratteristiche.
Il dizionario associato ad ogni modello ha i seguenti campi:

    'dropout_rate': float - rateo di "spegnimento" per il primo livello di dropout
    'dropout_rate_1': float - rateo di "spegnimento" per il secondo livello di dropout
    'lr': float - learning rate
    'layer': int - numero di neuroni del primo layer denso
    'layer_1': int - numero di neuroni del primo layer denso
    'layer_2': int - numero di neuroni del primo layer denso
    'score_function': function - funzione di distanza selezionata per la rete
    'margin': float - margine da utilizzare per la funzione di loss contrastiva
    'fourth_layer': boolean - indica se aggiungere un quarto layer con 512 neuroni attivazione sigmoidale 
	'batch_size': int - batch size utilizzata per l'apprendimento

### Pseudo-etichette
Ciascun file di pseudo-etichette ricavato viene salvato come file .pickle contenente un dizionario serializzato, contenente i seguenti campi:

    'threshold': float - valore di soglia da utilizzare per convertire le distanze in pseudo-etichette
    'distances': array monodimensionale di float - mappa linearizzata delle distanze calcolate 
    'shape': tupla bidimensionale di int - dimensione originale della coppia di immagini
Con i campi restituiti è possibile quindi ricostruire in un secondo momento la mappa delle pseudo-etichette dopo aver ricaricato il file. Quest'ultimo deve avere lo stesso nome della coppia di immagini a cui fa riferimento.

### Clustering
Per ogni coppia di immagini viene salvato un file .pickle contenente un dizionario di clusters serializzato, la cui struttura è del tipo:
    
    {indice di cluster: set di indici di pixel appartenenti al medesimo cluster}
Il dizionario ha tante chiavi quanti sono i cluster calcolati per ogni coppia.

### Valutazione

Per ogni sperimentazione, sulla base dei dati raccolti, sono stati stipulati i seguenti file:

    nomemodello_stats.csv
    # report contenente metriche e informazioni sull'apprendimento con l'ottimizzazione degli iperparametri (trials di hyperas)
    
    dataset[_immagine]_on_nomemodello_tipoesecuzione.csv
    # report contenente metriche sul test del dataset "dataset" sul modello "nomemodello" eventualmente senza (no fine tuning) 
    # o con fine tuning con tutte le pseudo-etichette (all), una percentuale (x%) o estratte per vicinato (raggio y)

    dataset[_immagine]_on_nomemodello_tipoesecuzione.png
    # immagine contenente la mappa del cambiamento inferita, la stessa con i pixel etichettati in evidenza e la ground truth.

    dataset[_immagine]_on_nomemodello_tipoesecuzione_corrected.png
    # come la precedente, ma con la correzione spaziale applicata

    dataset[_immagine]_on_nomemodello_tipoesecuzione_heatmap.png
    # mappa di calore sulle distanze inferite con la rete

Inoltre, sono anche presenti file riguardanti la generazione delle pseudo-etichette:

    dataset_nomeimmagine_[ED/SAM]_pseudo_rescaling_[True/False].csv
    # report contenente le metriche sulle pseudoetichette dell'immagine "nomeimmagine" in dataset con la distanza indicata e l'eventuale rescaling
    
    dataset_nomeimmagine_[ED/SAM]_pseudo_rescaling_[True/False].png
    # immagine contenente la pseudo-mappa del cambiamento, la stessa con i pixel etichettati in evidenza e la ground truth.
    
    dataset_nomeimmagine_[ED/SAM]_pseudo_rescaling_[True/False]_corrected.csv
    # come la precedente, ma con la correzione spaziale applicata

    dataset_nomeimmagine_x%.png
    # mappe contenenti l'x% relativo delle pseudo-etichette cambiamento e non cambiamento sia su mappe singole che su una combinata

    dataset_nomeimmagine_radiusy.png
    # mappe contenenti le pseudo-etichette cambiamento e non cambiamento sia su mappe singole che su una combinata, estratte per vicinato con raggio y

    

## Installazione

    pip install -r requirements.txt

**Python  3.9**

Packages:

* [Tensorflow 2.7.0](https://www.tensorflow.org/) 
* [Keras 2.7.0](https://github.com/keras-team/keras)
* [Matplotlib 3.5.0](https://matplotlib.org/)
* [Scipy 1.7.3](https://www.scipy.org/)
* [Numpy 1.21.5](https://www.numpy.org/)
* [Scikit-learn 1.0.1](https://scikit-learn.org/stable/)
* [Scikit-image 0.18.3](https://scikit-image.org/)
* [Hyperas 0.4.1](https://github.com/maxpumperla/hyperas)
* [Hyperopt 0.2.5](https://github.com/hyperopt/hyperopt)
* [Pillow 8.4.0](https://pillow.readthedocs.io/en/stable/)



## Utilizzo

Per configurare l'esecuzione dell'algoritmo è necessario impostare i parametri dell'area *settings* di net.conf. Qui di seguito si indica il significato dei vari campi: 

    [setting]
    train_set               # nome del dataset per il training. Tale nome deve averlo anche la sezione con le informazioni riguardo ad esso
    test_set                # nome del dataset per il training. Tale nome deve averlo anche la sezione con le informazioni riguardo ad esso
    distance                # ED=> utilizzo della distanza euclidea, SAM => utilizzo della distanza SAM. 
    model_name              # nome del modello da salvare/caricare
    apply_rescaling         # True=>applica rescaling minmax ai dati  False=> Non applica il rescaling ai dati
    operation               # 0 => Avvia la procedura di apprendimento della rete, 1 => Avvia la procedura di clustering, 2 => Avvia la procedura di testing
    fine_tuning             # -1 => non applica ft, 0 => ft con tutte le pseudo, 1 => ft con selezione per percentuale, 2 => ft con selezione per vicinato, 3 => ft con selezione per percentuale delle pseudo + etichette reali per coppie di pixel scartate, 4 => ft con utilizzo di sole etichette reali per coppie di pixel scartate dalla selezione per percentuale, 5 => ft con selezione per vicinato delle pseudo + etichette reali per coppie di pixel scartate, 6 => ft con utilizzo di sole etichette reali per coppie di pixel scartate dalla selezione per vicinato, 7 => active learning con uncertainty sampling, 8 => active learning con k-means ed estrazione randomica dei pixel per ogni cluster, 9 => active learning con k-means ed estrazione dei pixel più incerti per ogni cluster
    pseudo_percentage       # valore in [0,1] indica la percentuale di pseudo-etichette da estrarre
    pseudo_radius           # intero positivo, indica il raggio entro il quale considerare il vicinato per l'estrazione delle etichette
    uncertainty_percentage  # percentuale di esempi incerti da selezionare per l'active learning con uncertainty sampling
    kmeans_percentage       # percentuale di esempi da scegliere per ogni cluster con algoritmo K-Means

Oltre alla configurazione dei valori per l'esecuzione, è necessario anche inserire le informazioni relative al dataset che si intende utilizzare:

    [nome_dataset]
    imgAPath          # path in cui inserire le immagini cronologicamente precendenti
    imgAPath          # path in cui inserire le seconde immagini cronologicamente successive
    labelPath         # path in cui inserire le etichette per le coppie di immagini
    pseudoPath        # path in cui vegnono salvate/caricate le pseudo-etichette per il fine tuning
    matLabel          # nome dell'etichetta del campo contenente i dati (solo per file .mat)
    changedLabel      # etichetta che nella ground truth indica una coppia cambiata
    unchangedLabel    # etichetta che nella ground truth indica una coppia non cambiata
    unknownLabel      # etichetta che nella ground truth indica una coppia ignota

NB: i dataset devono essere organizzati in modo che la quadrupla (immagine "prima", immagine "dopo", etichette, pseudo etichette)
siano quattro file (o cartelle con le immagini da comporre) con lo stesso nome nei path indicati dalla sovracitata sezione.

NB2: il dataset *ONERA TEST* presenta un path in più chiamato "clusterPath", in cui inserire i dizionari di clusters serializzati per le coppie di immagini.

Inoltre, è anche possibile cambiare alcuni parametri dell'algoritmo di ricerca automatica Hyperas in base alla funzione distanza:

    [hyperas settings sigla_distanza]
    batch_size      # lista (scritta in sintassi python) dei possibili valori di batch size utilizzabili
    max_dropout     # float in ]0,1[ che indica il limite superiore dell'intervallo in cui scegliere il dropout rate
    neurons         # lista (scritta in sintassi python) dei possibili numeri di neuroni assegnabili al primo livello
    neurons_1       # lista (scritta in sintassi python) dei possibili numeri di neuroni assegnabili al secondo livello
    neurons_2       # lista (scritta in sintassi python) dei possibili numeri di neuroni assegnabili al terzo livello
    fourth_layer    # True=>aggiunta del quarto livello a 512 neuroni False=>tale livello non viene aggiunto

Una volta impostati tutti i valori necessari all'interno del file di configurazione, è possibile avviare il codice contenuto in main.py per avviare
la fase di addestramento, clustering o predizione, il codice contenuto in predutils.py per generare le pseudo-etichette o i "labelby" per la generazione delle mappe.

### Addestramento di un modello

Impostare il valore *operation = 0*, i valori di *train_set* e *test_set* con i nomi dei rispettivi dataset che si intende utilizzare (attualmente sono implementati  *"ONERA TRAIN"* e *"ONERA TEST"*), il valore di *distance* che indichi la funzione di distanza selezionata (tra "*ED*" e "*SAM*"), il valore *model_name=modello_scelto* e il valore di *apply_rescaling* a *True* (consigliato) o *False*. 
Modificare, inoltre, se lo si ritiene opportuno i valori delle sezioni *hyperas settings SAM/ED*.
Dopodichè è possibile lanciare main.py senza altri argomenti

    python3 main.py

In questo modo verrà eseguito il caricamento e il pre-processing del dataset, l'apprendimento e della rete sui set indicati con l'ottimizzazione degli iperparametri e successivo test. In particolare, verranno ottimizzati il numero di neuroni per layer (nei range indicati), il learning rate, i dropout rate e la batch size. Il migliore modello viene selezionato in base alla minore *loss* registrata sul validation set (default=20% del train set) Al termine dell'apprendimento un report dei vari trials verrà salvato in *stat* e il modello migliore risultante in *model* (entrambi i path sono indicati in config.py)

### Testing di un modello

Impostare il valore *operation = 2*, il valore di *test_set* con il nome del dataset da utilizzare per il testing, il valore di *model_name* col nome del modello presente in *model* da testare e il valore di *apply_rescaling*
a *True* (consigliato) in caso si voglia applicare il rescaling sul dataset di input, altrimenti a *False*.
In caso si voglia attuare il testing con fine tuning, si imposti il valore di *fine_tuning* a:
* 0 per utilizzare tutte le pseudo-etichette, 
* 1 per utilizzare la selezione per percentuale (e assegnando a *pseudo_percentage* la quantità desiderata),
* 2 per utilizzare la selezione per raggio (e assegnando a *pseudo_radius* il raggio desiderato),
* 3 per utilizzare la selezione per percentuale delle pseudo etichette + etichette reali per le coppie di pixel scartate (e assegnando a *pseudo_percentage* la quantità di pseudolabel desiderata),
* 4 per utilizzare solo le etichette reali per gli esempi scartati dalla selezione per percentuale (e assegnando a *pseudo_percentage* la quantità di pseudolabel da scartare)
* 5 per utilizzare la selezione per raggio delle pseudo etichette + etichette reali per le coppie di pixel scartate (e assegnando a *pseudo_radius* il raggio desiderato per le pseudolabel)
* 6 per utilizzare solo le etichette reali per gli esempi scartati dalla selezione per raggio (e assegnando a *pseudo_radius* il raggio desiderato in cui scartare le pseudolabel)
* 7 per utilizzare active learning con uncertainty sampling (e assegnando a *uncertainty_percentage* la quantità di esempi incerti a cui assegnare le etichette reali)
* 8 per utilizzare active learning con k-means ed estrazione randomica dei pixel per ogni cluster (e assegnando a *kmeans_percentage* la quantità di esempi randomici a cui assegnare le etichette reali)
* 9 per utilizzare active learning con k-means ed estrazione dei pixel più incerti per ogni cluster (e assegnando a *kmeans_percentage* la quantità di esempi più incerti a cui assegnare le etichette reali)

Altrimenti, per eseguire il testing senza tuning, si imposti *fine_tuning=-1*

Dopodichè si può lanciare nuovamente main.py con il comando precedente.

In questo modo sarà avviato il caricamento e pre-processing del test dataset, il caricamento del modello, l'applicazione del fine tuning (se richiesta) e, infine, l'esecuzione della predizione sul test set indicato.
In *stat* verranno generati un file .csv contenente le statistiche relative al testing con e senza correzione spaziale, una mappa del calore che rappresenta le distanze calcolate dalla rete per la coppia di immagini e due immagini contenenti le mappe predette (con e senza correzione). Queste due presentano la mappa predetta (*Total Prediction*), la stessa con una maschera che copra i pixel ignoti (*Comparable Preditiction*) e la *ground truth*. Il codice cromatico stabilisce che i pixel rossi indichino cambiamento, quelli blu non-cambiamento e quelli gialli non posseggono verità di fondo. 

### Generazione dei cluster
Impostare il valore *operation = 1*, il valore di *test_set* con il nome del dataset da utilizzare per generare i cluster, il valore di *apply_rescaling* a *True* o *False*, in caso si voglia o meno applicare il rescaling sul dataset in input.

A questo punto sarà possibile lanciare main.py con il comando precedente.

I dizionari di clusters saranno salvati nella sottocartella "cluster" della cartella "oneratest".

### Generazione delle pseudo-etichette
Impostare il valore di *train_set* con il nome del dataset da utilizzare per calcolare le pseudo-etichette, il valore di *distance* con "*ED*" o "*SAM*" in base alla funzione di distanza che si vuole utilizzare per il calcolo, il valore di *apply_rescaling* a *True* o *False*, in caso si voglia o meno applicare il rescaling sul dataset in input. 
A questo punto sarà possibile lanciare predutils.py senza altri argomenti
    
    python3 predutils.py
Le pseudo etichette verranno generate applicando la distanza scelta sulla coppia di immagini e ricavando la soglia di conversione con il metodo Otsu. Saranno poi salvate nel percorso "pseudoPath" del dataset, ciascuna con il nome della coppia di immagini di riferimento, come file .pickle. In *stat* inoltre verranno raccolte le mappe e le statistiche relative alle pseudo-etichette, con e senza correzione spaziale (raggio di default=3).

### Generazione dei plot di pseudo-etichette
In caso si voglia visualizzare la distribuzione delle etichette estratte con i metodi per vicinato o per percentuale, si imposti il valore di *train_set* con il nome del dataset di cui stampare le pseudo etichette. Poi se si vuole un plot delle mappe estratte per percentuale, si lanci labelbyperc_generation.py:

    python3 labelbyperc_generation.py
e in *stat* verranno generate tutte le mappe contenenti dal 10% al 90% delle migliori pseudo-etichette, con una differenza del 10% tra una immagine e l'altra.
<br>Diversamente per un plot delle mappe estratte per raggio, si imposti *pseudo_radius* con il valore massimo di raggio da considerare per il plotting, e si lanci labelbyneigh_generation.py:

    python3 labelbyneigh_generation.py

In *stat* verranno generate tutte le mappe ottenute con estrazione per vicinato dal raggio 2, al raggio impostato.<br>
Il formato di entrambe le immagini comprende una mappa con le sole etichette di cambiamento (*C label*), una con le sole etichette di non cambiamento (*N label*) e una con entrambe (*NC label*). Il giallo indica i pixel esclusi.


#### Preprocessing

Durante la fase di preprocessing i dati contenuti nel dataset vengono organizzati in array triple (pixel A, pixel B, etichetta), normalizzati in [0, 1] attraverso MinMax, e le etichette subiscono un refactoring dei valori in modo da essere sempre logicamente consistenti con gli algoritmi utilizzati (i.e. i pixel cambiati vengono etichettati con "0", quelli non cambiati con "1" e quelli ignoti con "2"). Inoltre, le immagini vengono anche ripulite da eventuali coppie non etichettate (in caso di training).
Questa fase viene attuata prima di ogni fase di training o testing.


#### Valutazione 

La valutazione finale delle prestazioni dei singoli modelli viene effettuata mediante matrice di confusione costruita attraverso le etichette reali degli esempi, caricate assieme alle coppie, e le predizioni della Rete siamese. I valori riportati come "veri positivi" sono i pixel cambiati correttamente predetti, i "veri negativi" sono i pixel non cambiati correttamente predetti, i "falsi positivi" sono i pixel non cambiati predetti come cambiati e i "falsi negativi" sono i pixel cambiati predetti come non cambiati. In caso di test, la matrice viene costruita sia prima che dopo l'esecuzione della correzione spaziale (con raggio di default=3) sulle etichette predette dalla rete, in modo da poter calcolare l'*Overall Accuracy* in entrambi i momenti sia rispetto alla ground truth, sia rispetto alle pseudo-etichette, ove utilizzate. Diversamente per il valore calcolato sul *validation set* o sul *test set durante il training*, il calcolo viene eseguito senza la correzione spaziale.

Un altro criterio di valutazione è dato dal tempo impiegato nell'esecuzione dei vari passaggi, registrato sempre in secondi. Nei file .csv ottenuti dal training è presente il campo *time* che indica il tempo impiegato per l'esecuzione della singola *run*.
Nei file .csv delle pseudo etichette sono presenti *generation_time* e *correction_time* che indicano rispettivamente i tempi di generazione e di applicazione della correzione spaziale.
Nei file. csv ottenuti dal testing sono presenti i campi:
    
    prediction_time: tempo impiegato dalla sola predizione
    extraction_time: tempo impiegato per l'esecuzione di uno dei due algoritmi di selezione.
                     Per la selezione per vicinato viene compreso anche il tempo per eseguire 
                     la correzione spaziale
    correction_time: tempo impiegato per la correzione spaziale sulla mappa predetta
    ft_time:         tempo impiegato per l'esecuzione del fine tuning

#### File necessari all'utilizzo degli script

Per rendere possibile l'esecuzione del training o la generazione delle pseudo-etichette sono necessari i seguenti file:

* Il dataset contenente le immagini divise in cartelle tra "prima", "dopo" ed "etichette"  in *data/nome_dataset/*

Per l'esecuzione del testing invece sono in più necessari i seguenti file:
* Il modello da testare all'interno di *model*, come coppia di file *nomemodello.h5* e *nomemodello_param.pickle*.
* Qualora si voglia applicare il fine tuning, le pseudo etichette come file *nomeimmagine.pickle* nella cartella dedicata alle pseudo-etichette del dataset di riferimento.
* Qualora si voglia applicare la configurazione 8 o 9 di fine tuning, i dizionari di cluster come file *nomeimmagine.pickle* nella cartella dedicata al clustering del dataset di riferimento.

I file delle pseudo etichette sono necessari anche in caso si voglia eseguire il plot con gli script *"labelby"*.

### Script e funzioni principali
Si propone una panoramica degli script e delle funzioni principali. Per maggiori dettagli, riferirsi ai commenti nel codice.

#### config.py
Questo file contiene tutte le costanti "interne" al programma, come le label utilizzate, il margine per la loss, lo split per il *validation*, i percorsi per il file di configurazione, quello per il salvataggio dei modelli e quello per il salvataggio dei log. Contiene anche le variabili utilizzate per passare i dataset e le impostazioni alla funzione che esegue la ricerca degli iperparametri.

#### dataprocessing.py
Questo modulo contiene le funzioni necessarie al caricamento e al preprocessing del dataset.<br> La funzione **load_dataset(name, conf)** permette di caricare un dataset con nome "name" (elencato tra quelli in *net.conf*) e con le configurazioni lette dal parser "conf" in input. Lo restituisce sottoforma di lista di immagini "prima", lista di immagini "dopo", lista delle etichette e lista dei nomi delle coppie. Attualmente, può solo caricare immagini in ".mat" e in “.tif” se decompresse (ovvero ogni immagine conta come un un vettore di pixel per una banda). Per caricare i file “.tif”, essi devono essere posti in cartelle con lo stesso nome delle etichette (compresi di estensione) e ciascuna banda deve avere il nome nel corretto ordine alfanumerico (i.e. prima banda => B01, seconda banda=>B02…).<br> 
La funzione **preprocessing(..., conf_section, keep_unlabeled, apply_rescaling)** prende in input i risultati di *load_dataset* (...) e ,utilizzando le informazioni passate dal parser *conf_section*, esegue il pre-processing del dataset. Ciò viene fatto linearizzando le immagini, eseguendo il MinMax sulla concatenazione di queste (se *apply_rescaling=True*), cambiando le etichette e generando i due array delle coppie di pixel e le rispettive etichette, eventualmente rimuovendo quelle ignote (se *keep_unlabeled*=False). Il risultato è un unico array contenente tutte le coppie di pixel di tutte le immagini, assieme al corrispettivo array delle etichette.<br>
La funzione **generate_pca_dataset(pairs)** crea un nuovo dataset che contiene le bande spettrali concatenate della prima e della seconda immagina della coppia in esame. In seguito, a questo dataset viene applicato l'algoritmo PCA (Principal Component Analysis) che riduce le dimensioni del dataset a 2 componenti principali.

#### labelbyneigh_generation.py
Script per la generazione dei plot delle etichette estratte per vicinato con raggio variabile. Fa utilizzo della funzione di estrazione inserita in predutils.py

#### labelbyperc_generation.py
Script per la generazione dei plot delle etichette estratte per vicinato con percentuale variabile. Fa utilizzo della funzione di estrazione inserita in predutils.py per ottenere le coppie di pixel in ordine di distanza (crescente per quelle cambiate, decrescente per quelle non cambiate) e genera poi i plot da esse.

#### main.py
Script principale per l'esecuzione del training o del testing di un modello. Il suo utilizzo è già stato discusso nel corso di questa guida.

#### predutils.py
Modulo contenente le funzioni di utility per la predizione, nonchè lo script di generazione delle pseudo-etichette.<br>
La funzione **spatial_correction(prediction, radius)** esegue la correzione spaziale sulla predizione in input (già riorganizzata in un array bidimensionale), facendo scorrere un kernel di raggio *radius* e riassegnando ad ogni pixel la classe predominante all'interno di esso. In caso di parità, si mantiene la classe originale. <br>
La funzione **pseudo_labels(first_img, second_img, dist_function, return_distances)** permette di ricavare le pseudo etichette dalla coppia di immagini *(first_img, second_img)* con la funzione di distanza *dist_function*. Ciascuna immagine ha 2 dimensioni (altezza x larghezza, bande spettrali). Si può decidere di ottenere direttamente la mappa delle pseudo etichette e la soglia ricavata ponendo *return_distances=False*, oppure ottenere la mappa delle distanze, ponendo *return_distances=True*. <br>
La funzione **pseudo_by_percentage_sam(pseudo_dict, percentage)** estrae da una mappa di pseudo etichette *pseudo_dict* in formato dizionario (vedi sezione Pseudo-etichette) una percentuale indicata da *percentage* (float in ]0, 1]) delle migliori etichette. La misura di bontà considerata è la distanza tra i due pixel: tanto più è estrema, tanto più sarà certa la classificazione. La funzione, quindi, ordina i pixel cambiati in maniera decrescente in base alla distanza e quelli non cambiati in maniera crescente. Dopodichè viene eseguito un "taglio" e restituiti due array contenenti le posizioni della migliore percentuale coppie e le relative etichette.<br>
La funzione **pseudo_plus_labels_by_percentage_sam(pseudo_dict, percentage, img_label)** è utilizzata per applicare la configurazione 3 di fine tuning.<br>
La funzione **labels_by_percentage_sam(pseudo_dict, percentage, img_label)** è utilizzata per applicare la configurazione 4 di fine tuning.<br>
La funzione **pseudo_by_neighborhood(pseudo_dict, radius)** estrae da una mappa di pseudo etichette *pseudo_dict* in formato dizionario (vedi sezione Pseudo-etichette) le migliori etichette. La misura di bontà considerata è la presenza pixel diversi in un vicinato quadrato di raggio *radius*. Se è presente almeno un pixel di classe diversa, il pixel centrale viene scartato. La funzione quindi scandisce l'immagine, rimuove i pixel spuri e restituisce due array contenenti le posizioni delle migliori coppie e le relative etichette.<br>
La funzione **pseudo_plus_labels_by_neighborhood(pseudo_dict, img_label, radius)** è utilizzata per applicare la configurazione 5 di fine tuning.<br>
La funzione **labels_by_neighborhood(pseudo_dict, img_label, radius)** è utilizzata per applicare la configurazione 6 di fine tuning.<br>
La funzione **labels_by_percentage_uncertainty(model, pairs, img_label, percentage)** è utilizzata per applicare la configurazione 7 di fine tuning.<br>
La funzione **labels_by_percentage_k_means_random(img_label, percentage, cluster_path)** è utilizzata per applicare la configurazione 8 di fine tuning.<br>
La funziome **labels_by_percentage_k_means_uncertainty(img_label, percentage, cluster_path, model, pairs)** è utilizzata per applicare la configurazione 9 di fine tuning.

#### siamese.py
Questo modulo contiene tutte le funzioni principali utili alla costruzione, apprendimento e *fine tuning* del modello di Rete Siamese.<br>
La funzione **hyperparam_search(train_set, train_labels, test_set, test_labels, distance_function, name, hyperas_search)** permette di effettuare il training con ottimizzazione degli iperparametri attraverso Hyperas sul train set indicato, con la funzione di distanza e le impostazioni passate in input. Viene anche eseguito, ad ogni iterata, un test sul set indicato e al termine dell'apprendimento la funzione salva le statistiche dei vari trial in un file .csv e il miglior modello risultante . Esso è scelto in base al più basso valore di loss sul *validation set*. Il parametro "name" indica il nome da dare al modello salvato.<br>
La funzione **siamese_model(train_set, train_labels, test_set, test_labels, score_function)**
è quella che effettua la costruzione, il training e il testing del modello per ciascuna iterata Hyperas. Restituisce infatti un dizionario contenenti le metriche registrate nella run corrente, secondo la sintassi di Hyperas (Si rimanda alla documentazione della libreria per maggiori info). L'apprendimento viene eseguito su un massimo di 150 iterate con un *EarlyStopping callback*, ovvero il training si interrompe se entro un numero di epoche (impostato a 10) la metrica tenuta sotto controllo (la loss sul *validation set*) esibisce miglioramenti.<br>
La funzione **build_net(input_shape, parameters)** costruisce e compila un modello di rete neurale utilizzando le librerie funzionali di Keras. La funzione viene sia utilizzata in fase di training che in fase di caricamento di un modello salvato. In input richiede la *shape* del singolo input e un dizionario contenente i parametri da applicare per la costruzione. I contenuti del dizionario sono descritti nella sezione **Modello**.
La funzione **fine_tuning(model, batch_size, x_retrain, pseudo_labels)** si occupa di ri-eseguire il training del modello *model*, sul dataset e le pseudo-etichette passati in input. Restituisce, oltre al modello ri-addestrato, i valori di loss su *train e valdation set*, l'accuracy sul *validation*, il numero di epoche e il tempo impiegato per il *re-train*.

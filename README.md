# ProjetGazouillis

## Streaming

Avant de lancer un stream, il faut ajouter un fichier "twitter_config.py" à la racine du projet, qui contient les lignes :

consumer_key = 'YOUR-CONSUMER-KEY'

consumer_secret = 'YOUR-CONSUMER-SECRET'

access_token = 'YOUR-ACCESS-TOKEN'

access_secret = 'YOUR-ACCESS-SECRET'

Pour obtenir ces clés/tokens, des infos sont dispos ici : https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/


## Training

Réservation : https://docs.google.com/spreadsheets/d/1w5O_sSe55dsKJ__VjtUd0El3Vew17k76sfv7khszmMo/edit?ts=573365f3#gid=0

Commande : THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python base_script.py

ATTENTION : bien spécifier la bonne carte graphique (device=gpu0 ou gpu1 ou gpu2...) en se fiant à ce qui est écrit sur la Google Sheet.

### Compte bergegu

Password : cmZ=mhTk)fJvf*93

PATH_DATA = "/data/lisa/exp/bergegu/gazouillis/data/dataset.npy"

PATH_EXPERIMENT = "/data/lisa/exp/bergegu/gazouillis/experiments/new_name"

### Compte augustar

Password : upq_xehq!ay5b(GD

dataset.npy à transférer à chaque fois en local sur la machine réservée.

PATH_DATA = "/Tmp/augustar/data/dataset.npy"

PATH_EXPERIMENT = "/Tmp/augustar/experiments/new_name"


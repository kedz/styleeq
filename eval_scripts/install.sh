git clone https://github.com/tuetschek/e2e-metrics.git
cd e2e-metrics
pip install -r requirements.txt
pip install future

curl -L https://cpanmin.us | perl - App::cpanminus  # install cpanm
cpanm XML::Twig

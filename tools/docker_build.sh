#!/bin/bash
if [ ! -d "/home/jovyan/aicloud-examples" ] && [ ! -d "/home/jovyan/quick-start" ] && [ ! -f "/home/jovyan/.aicloud_example_flag" ]; then
        cp -R /tmp/aicloud-examples/quick-start /home/jovyan/
        touch /home/jovyan/.aicloud_example_flag
fi
if [ -d "/tmp/start" ]; then
        sh /tmp/start/*.sh
fi
if [ ! -f "/home/jovyan/.aicloud_ssh_flag" ]; then
        NAMESPACE=`echo "${NB_PREFIX}" | cut -d "/" -f 3`
        echo "n" | ssh-keygen -q -N "" -f ~/.ssh/${NAMESPACE}
        echo "IdentityFile ~/.ssh/${NAMESPACE}" >> ~/.ssh/config
        touch /home/jovyan/.aicloud_ssh_flag
fi
if [ ! -x "$(command -v nvidia-smi)" ]; then
        pip3 uninstall -y jupyterlab-nvdashboard
        jupyter labextension disable jupyterlab-nvdashboard
fi
jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.base_url=${NB_PREFIX} --NotebookApp.allow_origin='https://aicloud.sbercloud.ru,https://mlspace.aicloud.sbercloud.ru,https://test.aicloud.sbercloud.ru' --FileContentsManager.delete_to_trash=True

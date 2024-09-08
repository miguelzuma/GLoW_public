import os
import re

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    src_path = root+"/source"
    gen_path = src_path+"/generated"
    nb_path  = root+"/build/html/generated"

    all_files = sorted(os.listdir(nb_path))

    pattern = re.compile(r'^example.*\.html$')
    html_files = [f for f in all_files if pattern.match(f)]

    title = "Notebooks\n"\
            "=========\n"
    with open(src_path+"/notebooks.rst", 'w') as fp:
        fp.write(title)
        fp.write(".. toctree::\n    :maxdepth: 2\n")

    for f in html_files:
        base_name = os.path.splitext(f)[0]
        nb_name = base_name + '.ipynb'

        title = nb_name+'\n'+'='*len(nb_name)+'\n\n'

        text = '.. raw:: html\n\n'\
               '    <iframe src="%s" width="100%%" height="600"></iframe>\n' % f

        with open(gen_path+'/'+base_name+'_nb.rst', 'w') as fp:
            fp.write(title)
            fp.write(text)

        with open(src_path+"/notebooks.rst", 'a') as fp:
            fp.write('\n    generated/'+base_name+'_nb')

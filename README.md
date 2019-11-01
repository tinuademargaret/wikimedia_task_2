<h1>wikimedia_task_2</h1>
<h3>Repository for Wikimedia project of Outreachy 2019</h3>
<p>This task,exercises simple parsing of a Wikipedia article and classifying some of its sentences.</p>
<p>It achieves this in the following steps:
  <ol>
<li>Receives as input the title of a English Wikipedia article.
<li>Retrieves the text of that article from the MediaWiki API. If using Python, consider using python-mwapi for this.
<li>Identifies individual sentences within that text, along with the corresponding section titles. If using Python, mwparserfromhell can help you work with wiki markup.
<li>Runs those sentences through the model to classify them.
<li>Outputs the sentences, one per line, sorted by score given by the model.
<h3>System Requirement</h3><br>
<p>Python 2.7
  <h3>Installing dependencies</h3>
<p>Now, go to the repository folder and install the dependencies listed in the requirements.txt:<br>
  <code>(ENV) $ pip install -r requirements.txt</code>
<p>Download models from the link in the model/ folder
<p>Download dictionaries from the link in the embeddings/ folder
<p>To run the script, you can use the following command:<br>
<code>python classify.py -m models/model.h5 -v dicts/word_dict.pck -s dicts/section_dict.pck -o output_folder -l it</code>

<p>'-o', '--out_dir', is the output directory where we store the results

<p>'-m', '--model', is the path to the model which we use for classifying the statements.

<p>'-v', '--vocab', is the path to the vocabulary of words we use to represent the statements.

<p>'-s', '--sections', is the path to the vocabulary of section with which we trained our model.

<p>'-l', '--lang', is the language that we are parsing now, e.g. "en", or "it".

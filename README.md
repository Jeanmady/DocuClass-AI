python src/extraction.py

each box has one figure and one takeaway message.
must have numbers to back everrything up (all my claims)
lessons learnt, whats coming next, speculative last sentence


f1 macro of 0.809 and 98 perc accuracy baseline     
svm baseline: Fails on Environmental (0.48 Recall)
adding in gradient clipping and alpha smoothing (square root weighting)

ask all questions to the lllm and make it chose the closest one
with cloud llm cost benefits vs performance

week extension???

the measures i am using, macro measures, normalised is it imbalanced

see there is a problem. round 2! identify issue then try to address the issue ie the heavy issues with fire statemetns

 The Upgraded Baseline (The "Before")
What I did: Re-ran the SVM baseline using N-grams (capturing phrases, not just words).
Results: Got 98.25% Accuracy, but the Macro F1 was 0.81.
The Problem: I found that the SVM is "blind" to complex reports. It had a recall of only 0.48 for Environmental Statements, usually confusing them with Biodiversity reports.

3. The MiniLM Transformer (The "Middle")
Technical Fixes:
Implemented "Head-Tail" tokenization: Instead of just taking the start of the document, I’m stitching the first 255 and last 255 tokens together so the model sees the header and the conclusion/signatures.
Used Alpha-Weighted Focal Loss: Applied a massive penalty for missing small classes to handle that 188:1 imbalance.
Results: The MiniLM hit a Macro F1 of 0.84. It beat the SVM, but still made 14 errors in the test set.

4. The Hybrid Adjudicator (The "After" / Original Method)
What I did: Built a second tier using a local Mistral-Nemo (12B) model via Ollama.
The Logic: I took those 14 errors from the MiniLM and passed the text to Mistral along with the "Gold Standard" class definitions I generated.
The Result: Mistral rescued 13 out of the 14 errors.
Final Stats: The final hybrid system hit 99.85% Accuracy.

5. Data Insight found along the way
Fire Statements: Only 0.6% of the data were unreadable scans, but 60% of those scans were Fire Statements. It's a small gap, but it's in the most important category.

6. Where I am now
Final Result: 99.85% accuracy.
Remaining Error: I have exactly one document left that the system still can't get right (an Environmental Statement it thinks is Biodiversity).
Next step: Just focusing on the poster now.
Quick Summary Table (in case he wants to see it):
SVM: 0.81 F1 (Fails on technical reports)
MiniLM: 0.84 F1 (Handles context better, but still makes mistakes)
Hybrid (Final): 99.85% Accuracy (Rescued almost all mistakes using the 12B model)


1. On Result Presentation (Addressing Skepticism)
"I achieved 99.85% final accuracy with the Hybrid model, but I know markers can be skeptical of near-100% scores. On the poster, should I lead with that 99.85% headline, or should I prioritise the Macro F1 score (0.84 for local, ~0.97 for hybrid) to prove to them that the model isn't just ignoring the 188:1 class imbalance?"

2. On Technical Visuals (Addressing Marker 1 Feedback)
"Marker 1 noted a lack of schematic diagrams in my preliminary report. For the poster, I’ve designed a System Architecture diagram showing the 'Cascading' logic (MiniLM 
→
→
 Mistral-Nemo). Does the marking team prefer to see 'high-level flowcharts' for the showcase, or should I include technical details like the Head-Tail tokenization logic within that diagram?"

3. On Problem Justification (Addressing Marker 2 Feedback)
"Marker 2 felt the problem might be solvable by simple naming conventions. Should I include a specific 'Why AI is Mandatory' section on the poster—highlighting messy data like scan001.pdf and bundled documents—to shut down the 'naming convention' argument immediately?"

4. On the "Showcase" Demonstration (The 30% Mark)
"For the showcase day, the spec requires a 'running version' of the project. Since mine is a backend processing pipeline, would showing terminal logs of Mistral-Nemo reasoning be sufficient, or would you advise building a basic Streamlit UI to make the demonstration more interactive and accessible for the markers?"

5. On Qualitative Analysis (The 1st Class "Originality" Mark)
"I have exactly one remaining error where even the 12B model failed. To show 'Originality and Critical Analysis,' would it be a good idea to include a small 'Case Study of Failure' box on the poster to explain why that specific document is semantically impossible for the AI, or is it better to focus only on the successes?"

6. On Data Visualization
"To visualize the 188:1 imbalance, should I include a bar chart of all 23 classes, or is it more 'visually striking' to just compare the Top 5 and Bottom 5 classes in a simplified graphic to keep the poster from being too cluttered?"


cross validation

buiild prompr of the 5 classes filter out classes, way to find the threshold try validate by how many do i have to choose to get 99 to 100 naccuracy
how accurate ocr is accross document types ie fire statment

1. good way to do this for a company

2. people just dont do it, no compliance on the user behalf on floowing naming convention, could be paper docs etc 
prefixing or keep file nae excel or csv sheet to tell which one is it

5. 4 blocks problem interesint g gap how i sole problem then the result how good is it future work what do i need to tackle to make it ant bettereach one picture to make it better
self explanitory poster show flatmates

code no data
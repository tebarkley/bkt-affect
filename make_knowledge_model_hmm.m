function bnet = make_knowledge_model_hmm()

%define the connections within a timeslice
intra = zeros(2); %within slice connections
intra(1,2) = 1;
inter = zeros(2); %between slice connections
inter(1,1) = 1;

%define the number of hidden and observable states (assuming both are discrete)
K = 2; %num hidden states
Q = 2; %num obs states

ns = [K Q];
dnodes = 1:2; %discrete nodes
onodes = 2; %obsreved nodes

eclass1 = [1 2]; %the equivalence class that node i in slice 1 belongs to. 
eclass2 = [3 2]; %the equivalence class that node i in slice 2 belongs to. 
eclass = [eclass1 eclass2];

bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);


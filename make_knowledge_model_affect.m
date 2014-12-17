function bnet = make_knowledge_model_final()

%define the connections 
intra = zeros(3);
intra(2,3) = 1; %knowledge to question
intra(1,3) = 1; %affect to question
inter = zeros(3);
inter(2,2) = 1; %knowledge transitions

%define the number of hidden and observable states (assuming both are discrete)
A = 2; %num obs affect states
K = 2; %num of hidden knowledge states
Q = 2; %num obs question states

ns = [A K Q];
dnodes = 1:3; %discrete
onodes = [1 3]; %observed affect and question nodes

eclass1 = [1 2 3]; %the equivalence class that node i in slice 1 belongs to. 
eclass2 = [1 4 3]; %the equivalence class that node i in slice 2 belongs to. 
eclass = [eclass1 eclass2];

bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);


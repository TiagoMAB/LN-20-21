#!/bin/bash

mkdir -p compiled images

for i in sources/*.txt tests/*.txt; do
	echo "Compiling: $i"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt $i | fstarcsort > compiled/$(basename $i ".txt").fst
done

fstconcat compiled/horas.fst compiled/dois_pontos.fst > compiled/temp.fst
fstconcat compiled/temp.fst compiled/minutos.fst > compiled/text2num.fst
rm compiled/temp.fst
fstconcat compiled/horas.fst compiled/preencher_minutos.fst > compiled/temp.fst
fstunion compiled/text2num.fst  compiled/temp.fst > compiled/lazy2num.fst
rm compiled/temp.fst
fstproject --project_type=input compiled/horas.fst > compiled/temp1.fst
fstproject --project_type=input compiled/dois_pontos.fst > compiled/temp2.fst
fstconcat compiled/temp1.fst compiled/temp2.fst > compiled/temp.fst
fstunion compiled/quartos.fst compiled/meias.fst > compiled/temp3.fst
fstconcat compiled/temp.fst compiled/temp3.fst > compiled/rich2text.fst
rm compiled/temp*
fstcompose compiled/rich2text.fst compiled/lazy2num.fst > compiled/temp1.fst
fstunion compiled/lazy2num.fst compiled/temp1.fst > compiled/rich2num.fst
rm compiled/temp*
fstinvert compiled/text2num.fst > compiled/num2text.fst

for i in compiled/*.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
    fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done

for i in compiled/*A*.fst; do
  echo "Testing the transducer 'rich2num' with input $i"
  fstcompose $i compiled/rich2num.fst | fstshortestpath | fstproject --project_type=output | fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./syms.txt
done

for i in compiled/*B*.fst; do
  echo "Testing the transducer 'num2text' with input $i"
  fstcompose $i compiled/num2text.fst | fstshortestpath | fstproject --project_type=output | fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./syms.txt
done

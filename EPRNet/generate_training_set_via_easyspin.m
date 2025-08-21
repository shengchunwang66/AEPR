% This script generates EPR spectra using EasySpin for a set of spin parameters

clf; clear;
addpath(genpath(pwd));

% Load the spin parameters from a text file
a = readmatrix('/home/ubuntu/code/matlab_0320/generate_spin.txt');
save_path = '/home/ubuntu/data/spin_trapping/lwpp0_3/';

% Set the parameters for the EPR simulation
lwpp = 0.3;
freq = 9.8;

for  i = 1:size(a)
    if a(i,1) == 0
        xAN = a(i,1)+0.01;
    else
        xAN = a(i,1);
    end 
    
    if a(i,2) == 0
        xAH = a(i,2)+0.01;
    else
        xAH = a(i,2);
    end 

    if a(i,3) == 0
        xA0p5 = a(i,3)+0.01;
    else
        xA0p5 = a(i,3);
    end 

    if a(i,4) == 0
        xAN2 = a(i,4)+0.01;
    else
        xAN2 = a(i,4);
    end 

    if a(i,5) == 0
        xAH3 = a(i,5)+0.01;
    else
        xAH3 = a(i,5);
    end 

    if a(i,6) == 0
        xAH4 = a(i,6)+0.01;
    else
        xAH4 = a(i,6);
    end 

    % EPR simulation using EasySpin
    ge = 2.002;
    
    Exp.nPoints = 2000;
    Exp.mwFreq = freq; 
    Exp.Range = [342.5 357.5];
    
    Sys.g = ge;
    Sys.lwpp = lwpp;
    Sys.Nucs = '14N, 1H, 1H, 14N, 1H, 1H';
    Sys.n = [1 1 1 1 1 1]; 
    Sys.A = [xAN xAH xA0p5 xAN2 xAH3 xAH4]*2.802495;
    Sys.Weight = 1; 
    [x1, y1]= garlic(Sys,Exp);
    y1=y1/(max(y1)-min(y1));
    
    x = x1*10;
    data = [x(:) y1(:)];

    % Save the data to a text file
    file_name = ['Compound_' num2str(i) '_g=' num2str(ge) '_AN=' num2str(xAN) '_AH=' num2str(xAH) '_A0p5=' num2str(xA0p5) '_AN2=' num2str(xAN2) '_AH3=' num2str(xAH3) '_AH4=' num2str(xAH4) '_lwpp=' num2str(lwpp) '.txt']
    save([save_path file_name],'data','-ascii');
    
end

exit;

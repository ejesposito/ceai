% clase 3 ej 1N = 10000;rand("seed", 1234);cantidadVecesGanaJuan = 0;cantidadVecesGanaJuanCon5 = 0;% tiro dadofor i=1:N    % dado Juan    a = randi(6);        % dado Pedro    b = randi(6);        % gano Juan      if(a > b)          cantidadVecesGanaJuan = cantidadVecesGanaJuan + 1;            % gano Juan  con 5      if(a==5)              cantidadVecesGanaJuanCon5 = cantidadVecesGanaJuanCon5 + 1;            endif        endif    
endfor
% estimar probabilidad que Juan haya ganado con 5p = cantidadVecesGanaJuanCon5 / cantidadVecesGanaJuanp_teorico = 4/15
function [tr]=cell2core(tr,cc)
%
%   Return a tr-tensor from the list of cores
d = numel(cc);
r = zeros(d,1);
n = zeros(d,1);


for i=1:d
    r(i) = size(cc{i},1);
    n(i) = size(cc{i},2);
end;



tr.d = d;
tr.n = n;
tr.r = r;
tr.node = cc;


end
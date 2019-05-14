function pval = cumKendallTest(Q,pop,tail)
% A cumulative Mann-Kendall trend test.
% Calculating p-values for monotonic trend of ordinal data sequences 
% (proteins) by a compound test based on Kendall's rank correlation.
% Program by Steinar Thorvaldsen, steinar.thorvaldsen@uit.no, July 2005. 
% Ref.: The DeltaProt toolbox at http://services.cbu.uib.no/software/deltaprot/
% Last changes 23. Dec 2010.
% Requires Matlab 7.1 or newer, and Matlab Statistics toolbox.

% Use:  pval = cumKendallTest(Q,pop) 
% returns p-values for testing the hypothesis of no correlation,
% against the alternative that there is a non-zero correlation
% between the columns in the (MxN)-matrix Q and the M-vector pop.
% If pval is small, say less than 0.01,then the trend is considered
% significantly different from zero.
% The test treats NaNs in Q or pop as missing values, and removes them.
% The case N=1 is the classical Mann-Kendall test.

% Input:
% Q:    MxN-matris with M independent ovservation series
% pop:  M-vector with population membership defined by an ordinal variable
%       e.g. optimal growth temperature for each of the M observation series.
% tail: The alternative hypothesis against which to compute p-values.
%       Choices are:
%           TAIL        Alternative Hypothesis
%           --------------------------------------
%           'ne'        trend is not zero (default)
%           'gt'        trend is greater than zero
%           'lt'        trend is less than zero
% Output:
% pval: P-value. Remark: - or + sign used here to indicate direction of trend
%
% cumKendall computes p-values using large-sample approximations.
% Unless data is very short (small sample size), this approximation is adequate.
% When there are ties in the data, the null distribution of Kendall's test
% may not be symmetric.  Computing a two-tailed p-value in such cases is
% not well-defined, and p-values for the two-tailed test is computed by
% doubling the more significant of the two one-tailed p-values.

% Please, use the following reference:
% Thorvaldsen, S. , Fl? T. and Willassen, N.P. (2010) DeltaProt: a software toolbox 
% for comparative genomics. BMC Bioinformatics 2010, Vol 11:573.
% See http://www.biomedcentral.com/1471-2105/11/573

% Other references:
% Kendall, M. and Gibbons, J.D. (1990) Rank Correlation Methods, 5th ed.  
% Hollander, M. and Wolfe, D.A. (1999) Nonparametric Statistical Methods, 2nd ed.
% Matlab 7.0 Statistical toolbox. (2005). The MathWorks, Inc.
% Thorvaldsen, S., Ytterstad, E. and Fl? T. (2006) Property-dependent analysis of
% aligned proteins from two or more populations. Proceedings of the 4th Asia-Pacific 
% Bioinformatics Conference (Eds.: T. Jiang et al.). Imperial College Press, pp. 169-178.
% See http://www.comp.nus.edu.sg/~wongls/psZ/apbc2006/189.pdf
% Thorvaldsen, S. , Fl? T. and Willassen, N.P. (2010) DeltaProt: a software toolbox 
% for comparative genomics. BMC Bioinformatics 2010, Vol 11:573.
% See http://www.biomedcentral.com/1471-2105/11/573

% Validate the input parameters:
if nargin < 2 || isempty(Q) || isempty(pop)
    error('cumKendallTest:TooFewInputs', ...
          'Requires data matrix Q and row-vector pop')
end
[M,N] = size(Q);
if M~=length(pop)
    error('The number of rows in matrix Q and row-vector pop must match in size')
end

if nargin < 3
    tail ='ne'; %default value
else
    switch tail %validate the tail parameter:
    case {'ne' 'gt' 'lt'}
        % these are ok
    otherwise
        error('cumKendallTest:UnknownTail', ...
          'The ''tail'' parameter value must be ''ne'', ''gt'', or ''lt''.');
    end %switch
end %if

% Compute the Kendall statistics:
KN=0; %main variable for the cumulative Kendall statistics
varKN=0; %the statistical variance of KN
for n=1:N %loop through all the independent observations
    Qpop=cat(2,Q(:,n),pop); % join data in one (Mx2)-matrix 
    [K, varK] = Kendall(Qpop);
    KN=KN+K;
    varKN=varKN+varK;
end
pval = pvalKendall(KN, varKN, tail);
pval = pval * sign(KN); %sign used here to indicate direction of trend
end %function cumKendallTest
%==========================================================================

function [K, varK] = Kendall(XY)
% Computes Mann-Kendall's pairwise rank correlation statistics.
% The test is based on the linear correlation of the 
% "concordances": sign(X(i)-X(j))*sign(Y(i)-Y(j)), i<j, with an 
% adjustment for ties. 

[n,p] = size(XY);
% Observations must be made in pairs:
if p ~= 2
    K = NaN;
    varK = NaN;
    disp ('Kendall test: Observations should be made in pairs!');
    return
end

% Pre-processing data by removeing rows with missing values, so all columns 
% have the same length.
ok = ~any(isnan(XY),2);
if ~all(ok)
    XY = XY(ok,:);
    n = sum(ok);
end

% At least two different observations not from the same population needed to do anything:
if n < 2 || max(XY(:,1))==min(XY(:,1)) || max(XY(:,2))==min(XY(:,2))
    K = 0;
    varK = 0;
    return
end

ntemp = n*(n-1) ./ 2; % expressen to use later

% Compute Kendall's K-statistics:
X = XY(:,1);
Y = XY(:,2);
[Xrank, Xadj] = tiedrank(X,1); %function in Matlab statistical toolbox
[Yrank, Yadj] = tiedrank(Y,1);
K = 0;
for k = 1:n-1
    K = K + sum(sign(X(k)-X(k+1:n)).*sign(Y(k)-Y(k+1:n)));
end
      
ties = ((Xadj(1)>0) || (Yadj(1)>0));
% In the case of ties:
if ties
    varK = ntemp*(2*n+5)./9 - (Xadj(3) + Yadj(3))./18 ...
         + Xadj(2)*Yadj(2)./(18*ntemp*(n-2)) ...
         + Xadj(1)*Yadj(1)./ntemp;
else % No ties
    varK = ntemp*(2*n+5)./9;
end %if

% corr = K ./ sqrt((n2const - Xadj(1)).*(n2const - Yadj(1))); %Kendall's tau
end % of function Kendall

%==========================================================================

function p = pvalKendall(K, varK, tail)
% Tail probability for Kendall's K statistics.

% Without ties, K is symmetric about zero, taking on values in
% -n(n-1)/2:2:n(n-1)/2.  With ties, it's still in that range, but not
% symmetric, and can take on adjacent integer values.
% An exact test may be obtained by use of recursion to get the exact 
% permutation distribution (see the function corr.m in MatLab Statistical 
% toolbox and Kendall (1990)), but here we use the large-sample 
% approximation to normality.

if nargin < 2 || varK <= 0 || isempty(K) || isempty(varK) 
    p = NaN;
    %disp('Function pvalKendall in file cumKendallTest.m :TooFewInputs')
    return
end
if nargin < 3
    tail ='ne'; %default value
else
    switch tail %validate the tail parameter:
      case {'ne' 'gt' 'lt'}
        % these are ok
      otherwise
        error('pvalKendall in file cumKendallTest.m :UnknownTail', ...
          'The ''tail'' parameter value must be ''ne'', ''gt'', or ''lt''.');
    end %switch
end %if

switch tail
  case 'ne'
    p = normcdf(-(abs(K)-1) ./ sqrt(varK));
    p = min(2*p, 1); % Don't count continuity correction at center twice
  case 'gt' %
    p = normcdf(-(K-1) ./ sqrt(varK));
  case 'lt'
    p = normcdf((K+1) ./ sqrt(varK));
end

end % of m-file function pvalKendall
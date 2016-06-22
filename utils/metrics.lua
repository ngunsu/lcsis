require 'torch'
local argcheck = require 'argcheck'
local env      = require 'argcheck.env'

function env.istype(obj, typename)
  if typename == 'torch.Tensor' then
    return torch.isTensor(obj)
  else
    return type(obj) == typename
  end
end

metrics = {}

------------------------------------------------------------------------------------------------------------------------
-- Error rate at 95% Recall
------------------------------------------------------------------------------------------------------------------------
function metrics.error_rate_at_95recall(...)
  local check = argcheck{
     help=[[
     This function computes the error rate when the recall is equal to 95.
     ]],
     {name="labels", type='torch.Tensor', help="Binary labels (1 = positive)"},
     {name="scores", type='torch.Tensor', help="Classification score"},
     {name="descending", type='boolean', default=true, help="If true sort in descending order. False ascending"}
  }
  -- Progress arguments
  local labels, scores, descending = check(...)

  -- Set recall goal
  local recall_goal = 0.95

  -- Sort label-score tuples by the score
  local sorted_scores, sorted_index = torch.sort(scores, 1, descending)

  -- Sum the positives cases (label 1)
  local number_of_true_matches = labels:eq(1):sum()

  -- Compute the value of 95% recall
  local threshold_number = recall_goal * number_of_true_matches

  -- Prepare variables for search
  local tp = 0
  local count = 0

  -- Run until find 95 % recall
  for i=1, sorted_index:size(1) do
    count = count + 1.0
    if labels[sorted_index[i]] >0 then
      tp = tp + 1.0
    end
    if tp >= threshold_number then
      break
    end
  end
  return (count - tp) / count * 100.0
end

------------------------------------------------------------------------------------------------------------------------
-- Average precision
------------------------------------------------------------------------------------------------------------------------
function metrics.average_precision(...)
  local check = argcheck{
     help=[[
     This function computes the average precision (AUC under precision recall curve)
     ]],
     {name="labels", type='torch.Tensor', help="Binary labels (1 = positive)"},
     {name="scores", type='torch.Tensor', help="Classification score"},
     {name="descending", type='boolean', default=true, help="If true sort in descending order. False ascending"}
  }

  -- Progress arguments
  local labels, scores, descending = check(...)

  -- Sort label-score tuples by the score
  local sorted_scores, sorted_index = torch.sort(torch.Tensor(scores), 1, true)

  -- Compute precision recall curve
  local pp = {}
  local rr = {}
  local tp = 0.0
  local fp = 0.0
  pp[1] = 1.0
  rr[1] = 0.0
  for i=1, sorted_index:size(1) do
    if labels[sorted_index[i]] == 1 then
      tp = tp + 1.0
    else
       fp = fp + 1.0
    end
    if labels:eq(1):sum() == 0 then
     rr[i+1] = 0
   else
     rr[i+1] = tp/labels:eq(1):sum()
   end
   pp[i+1] = tp/(tp+fp)
  end

  -- Compute the average precision
  local ap = 0
  for i=2, #pp do
    ap =  ap + ((rr[i] - rr[i - 1]) * (pp[i-1] + pp[i]))/2.0
  end
  return ap
end

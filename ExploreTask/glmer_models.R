library('lme4')

#this is assuming data is already scaled/z transformed

model2<-glmer(formula = choice ~ V + (V|participant_id),  
            data = gershman_data,
            family = binomial('probit'))


model3<-glmer(formula = choice ~ VTU + (VTU|participant_id),  
            data=gershman_data, 
            family = binomial('probit'))

model4<-glmer(formula = choice ~ V + RU + (V + RU|participant_id),  
            data=gershman_data, 
            family = binomial('probit'))


model5<-glmer(formula = choice ~ V + RU + VTU + (V + RU + VTU|participant_id),  
            data=gershman_data, 
            family = binomial('probit'))
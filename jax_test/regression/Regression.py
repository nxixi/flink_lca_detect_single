from jax_python.aad import AbstractAbnormalDetect
import numpy as np
import json

class Regression(AbstractAbnormalDetect):

    def version(self):
        return "1.0.0"

    def configure(self, dict):
        self.config=dict

    def transform(self, *df):
        _df=df[0]
        x_label=self.config['X']
        y_label=self.config['Y']
        poly_n=self.config.get('poly_n',1)
        assign_x=self.config.get('assign_x',-1.0)
        assign_y=self.config.get('assign_y',-1.0)
        X=_df[x_label]
        Y=_df[y_label]
        y_fit=np.polyfit(X,Y,poly_n)
        X_predict=-1.0
        Y_predict=-1.0
        if assign_y==-1.0:
            pass
        else:
            coefficient=[]
            for i in range(len(y_fit)):
                coefficient.append(y_fit[i]) if i!=len(y_fit)-1 else coefficient.append(y_fit[i]-assign_y)
            root=np.roots(coefficient)
            if len(root)==0:
                print('Equqtion has no root')
            else:
                flag=False
                for j in range(len(root)):
                    r=root[j]
                    if not isinstance(r,complex) and r>=0:
                        flag=True
                        X_predict=r
                if not flag:
                    print('Equation has no real root')

        equation=np.poly1d(y_fit)
        Y_fit=equation(X)

        if assign_x==-1.0:
            pass
        else:
            Y_predict=equation(assign_x)
        _df['fit']=Y_fit
        _df['x_predict']=X_predict
        _df['y_predict']=Y_predict
        model={'fit':list(y_fit)}
        print(model)
        return _df,bytes(json.dumps(model), encoding='utf-8')

    def fields(self):
        return [('fit','float'),('x_predict','float'),('y_predict','float')]

    def mode_append(self):
        return True


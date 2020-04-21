from aiida.engine import calcfunction, WorkChain, ToContext, while_, if_
from aiida.orm import Int
from aiida.plugins import CalculationFactory

ArithmeticAddCalculation = CalculationFactory('arithmetic.add')


@calcfunction
def add(x, y):
    return x + y


class PolishWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(PolishWorkChain, cls).define(spec)
        spec.expose_inputs(ArithmeticAddCalculation, namespace='add')
        spec.input('z', valid_type=Int)
        spec.output('sum', valid_type=Int)
        spec.outline( 
            cls.setup,
            cls.arithmetic_add,
            while_(cls.should_iterate)(
                if_(cls.should_run)(
                    cls.add,
                ),
                cls.update_iteration,
            ),
        )

    def setup(self):
        self.ctx.iteration = 5

    def should_iterate(self):
        return self.ctx.iteration > 0

    def should_run(self):
        return self.ctx.iteration == 1

    def update_iteration(self):
        self.ctx.iteration -= 1

    def arithmetic_add(self):
        inputs = self.exposed_inputs(ArithmeticAddCalculation, namespace='add')
        return ToContext(add=self.submit(ArithmeticAddCalculation, **inputs))

    def add(self):
        self.out('sum', add(self.ctx.add.outputs.sum, self.inputs.z))

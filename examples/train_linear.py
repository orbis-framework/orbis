import orbis as ob

xs = [ob.Tensor(i) for i in range(5)]
ys = [ob.Tensor(2*i + 1) for i in range(5)]

model = ob.Linear(1, 1)
opt = ob.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    total = 0.0

    for x, y in zip(xs, ys):
        opt.zero_grad()

        y_pred = model(x)
        loss = ob.mse(y_pred, y)

        loss.backward()
        opt.step()

        total += loss.value

    print(
        f"epoch {epoch}: "
        f"loss={total:.4f}, "
        f"w={model.w.value:.4f}, "
        f"b={model.b.value:.4f}"
    )

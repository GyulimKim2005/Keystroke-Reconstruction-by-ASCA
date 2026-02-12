import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import os
from GaAN_node import SimpleKeyboardGaAN 

def main():
    SAVE_PATH = "/content/drive/MyDrive/src/processed_dataset.pt"
    
    if not os.path.exists(SAVE_PATH):
        print("데이터 파일이 없습니다. dataset_node.py를 먼저 실행하세요.")
        return

    # weights_only=False로 최신 버전 파이토치 에러 방지
    data_dict = torch.load(SAVE_PATH, weights_only=False)
    full_dataset = data_dict['dataset']
    
    train_size = int(0.8 * len(full_dataset))
    train_dataset, test_dataset = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleKeyboardGaAN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"학습 시작! 장치: {device}")

    for epoch in range(1,501):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 모델에 x, edge_index, 그리고 batch 정보를 같이 전달
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "gaan_node_model.pth")
    print("모든 과정이 완료되었습니다.")

if __name__ == "__main__":
    main()
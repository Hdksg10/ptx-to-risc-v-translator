# download ubuntu image

IMAGE_PATH="./disk"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Cannot find image from path $IMAGE_PATH"
else
    echo "Found image, running..."
    qemu-system-riscv64 -machine virt -m 4G -smp cpus=2 -nographic \
        -kernel /usr/lib/u-boot/qemu-riscv64_smode/u-boot.bin \
        -netdev user,id=net0,hostfwd=tcp::42203-:22,hostfwd=tcp::42204-:45021 \
        -device virtio-net-device,netdev=net0 \
        -drive file=disk,format=raw,if=virtio \
        -device virtio-rng-pci \
        -fsdev local,id=fsdev0,path=/home/tangyue/ai/shared,security_model=mapped \
        -device virtio-9p-pci,fsdev=fsdev0,mount_tag=sharedfolder  
fi

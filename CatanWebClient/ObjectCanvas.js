class ObjectCanvas {
    constructor(ctx, scale = {x: 1, y: 1}) {
        this.ctx = ctx;
        this.size = {x: ctx.canvas.width * scale.x, y: ctx.canvas.height * scale.y};
        this.center = {x: this.size.x / 2, y: this.size.y / 2};
        this.objects = [];
    }

    addObject(obj) {
        this.objects.push(obj);
    }

    draw() {
        for(var i = 0; i < this.objects.length; i++) {
            this.objects[i].drawObject(this.ctx, this.center, this.size);
        }
    }
}

class CanvasObject {
    constructor(px, py, sx, sy) {
        this.pos = {x: px, y: py};
        this.scale = {x: sx, y: sy};
        this.children = [];
    }

    addChild(child) {
        this.children.push(child);
    }

    draw(ctx, center, scale) {
        // Draw object
    }

    drawObject(ctx, center, size) {
        var objectCenter = {x: center.x + this.pos.x * size.x, y: center.y - this.pos.y * size.y};
        var objectSize = {x: this.scale.x * size.x, y: this.scale.y * size.y};

        this.draw(ctx, objectCenter, objectSize);
    }

    drawChildren(ctx, center, size) {
        for (var i = 0; i < this.children.length; i++) {
            this.children[i].drawObject(ctx, center, size);
        }
    }
}

class Square extends CanvasObject {
    constructor(px, py, sx, sy, color) {
        super(px, py, sx, sy);
        this.color = color;
    }

    draw(ctx, objectCenter, objectSize) {
        ctx.fillStyle = this.color;
        ctx.fillRect(objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
        this.drawChildren(ctx, objectCenter, objectSize);
    }
}

class TextObject extends CanvasObject {
    constructor(px, py, sx, sy, text, font = "Arial", color = "#000000") {
        super(px, py, sx, sy);
        this.text = text;
        this.font = font;
        this.color = color;
    }

    draw(ctx, objectCenter, objectSize) {
        ctx.font = `${objectSize.y}px ${this.font}`;
        ctx.fillStyle = this.color;
        var textWidth = ctx.measureText(this.text).width;
        textWidth = textWidth > objectSize.x ? objectSize.x : textWidth;
        ctx.fillText(this.text, objectCenter.x - textWidth / 2, objectCenter.y + objectSize.y / 2, objectSize.x);
        this.drawChildren(ctx, objectCenter, objectSize);
    }
}
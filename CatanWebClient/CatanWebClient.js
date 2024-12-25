class backgroundWater extends CanvasObject {
    constructor(px, py, sx, sy) {
        super(px, py, sx, sy);

        // preload image
        this.image = new Image();
        this.image.src = "images/water.png";
    }

    draw(ctx, objectCenter, objectSize) {
        ctx.drawImage(this.image, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
    }
}

class backgroundMap extends CanvasObject {
    constructor(px, py, sx, sy) {
        super(px, py, sx, sy);

        // preload image
        this.image = new Image();
        this.image.src = "images/background_map.png";
    }

    draw(ctx, objectCenter, objectSize) {
        ctx.drawImage(this.image, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
    }
}

class Hex extends CanvasObject {
    constructor(px, py, sx, sy, hexX, hexY) {
        super(px, py, sx, sy);
        this.hexCoords = {x: hexX, y: hexY};
        this.text = new TextObject(0, 0, 0.866, 0.15, hexX + ", " + hexY, "Arial", "#000000");
        this.addChild(
            this.text
        );

        const imageUrls = ["images/bos.png", "images/brick.png", "images/shcaap.png", "images/graan.png", "images/steen.png"];
        this.images = []
        for (var i = 0; i < imageUrls.length; i++) {
            var image = new Image();
            image.src = imageUrls[i];
            this.images.push(image);
        }
        console.log(this.images);

        this.overlayImage = new Image();
        this.overlayImage.src = "images/rollOverlay.png";

        this.selected = false;
        this.recource = 1;
    }

    draw(ctx, objectCenter, objectSize) {
        ctx.drawImage(this.images[this.recource], objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
        if (this.selected) {
            ctx.drawImage(this.overlayImage, objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
        }
    }

    playSelectAnimation() {
        this.selected = true;
        setTimeout(() => {
            this.selected = false;
        }, 250);
    }

    update() {
        getTileData(this.hexCoords.x, this.hexCoords.y)
        .then(data => {
            this.text.text = data.number;
            this.recource = data.tile_type;
        });
    }
}

class Corner extends CanvasObject {
    constructor(px, py, sx, sy, cornerX, cornerY) {
        super(px, py, sx, sy);
        this.player = 0;
        this.cornerX = cornerX;
        this.cornerY = cornerY;
        this.level = 0;

        const imageUrls = [null, "images/bos.png", "images/brick.png", "images/shcaap.png", "images/graan.png", "images/steen.png"];
        this.images = [null]
        for (var i = 1; i < imageUrls.length; i++) {
            var image = new Image();
            image.src = imageUrls[i];
            this.images.push(image);
        }

        const settlementImageUrls = [null, "images/settlement_red.svg", "images/settlement_blue.svg", "images/settlement_green.svg", "images/settlement_yellow.svg"];
        this.settlementImages = [null]
        for (var i = 1; i < settlementImageUrls.length; i++) {
            var image = new Image();
            image.src = settlementImageUrls[i];
            this.settlementImages.push(image);
        }

        const cityImageUrls = [null, "images/city_red.svg", "images/city_blue.svg", "images/city_green.svg", "images/city_yellow.svg"];
        this.cityImages = [null]
        for (var i = 1; i < cityImageUrls.length; i++) {
            var image = new Image();
            image.src = cityImageUrls[i];
            this.cityImages.push(image);
        }

    }

    draw(ctx, objectCenter, objectSize) {
        if (this.available) {
            ctx.fillStyle = "#00FF00";
            ctx.beginPath();
            ctx.arc(objectCenter.x, objectCenter.y, objectSize.x / 3, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        if (this.player == 0) {
            ctx.fillStyle = "#000000";
            ctx.beginPath();
            ctx.arc(objectCenter.x, objectCenter.y, objectSize.x / 6, 0, 2 * Math.PI);
            ctx.fill();
            this.drawChildren(ctx, objectCenter, objectSize);
        }else {
            if (this.level == 1) {
                ctx.drawImage(this.settlementImages[this.player], objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            } else if (this.level == 2) {
                ctx.drawImage(this.cityImages[this.player], objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            }
        }
    }

    inClickBounds(clickPos, objectCenter, objectSize) {
        return Math.sqrt((clickPos.x - objectCenter.x) ** 2 + (clickPos.y - objectCenter.y) ** 2) < objectSize.x / 3;
    }

    onClick() {
        let newLevel = this.level + 1;
        if (newLevel > 2) {
            newLevel = 1;
        }
        setSettlement(this.cornerX, this.cornerY, game.player, newLevel)
        .then(data => {
            game.updateObject()
        }
        )
    }
}

class Road extends CanvasObject {
    constructor(px, py, sx, sy, roadX, roadY, rotation = 0) {
        super(px, py, sx, sy);
        this.player = 0;
        this.roadX = roadX;
        this.roadY = roadY;
        this.rotation = rotation;
        const imageUrls = [null, "images/road_red.svg", "images/road_blue.svg", "images/road_green.svg", "images/road_yellow.svg"];
        this.images = [null]
        for (var i = 1; i < imageUrls.length; i++) {
            var image = new Image();
            image.src = imageUrls[i];
            this.images.push(image);
        }
    }

    draw(ctx, objectCenter, objectSize) {
        if (this.available) {
            ctx.fillStyle = "#00FF00";
            ctx.beginPath();
            ctx.arc(objectCenter.x, objectCenter.y, objectSize.x / 3, 0, 2 * Math.PI);
            ctx.fill();
        }
        if (this.player == 0) {
            ctx.fillStyle = "#000000";
            ctx.beginPath();
            ctx.arc(objectCenter.x, objectCenter.y, objectSize.x / 10, 0, 2 * Math.PI);
            ctx.fill();
            this.drawChildren(ctx, objectCenter, objectSize);
        }else {
            ctx.save();
            ctx.translate(objectCenter.x, objectCenter.y);
            ctx.rotate(this.rotation);
            ctx.translate(-objectCenter.x, -objectCenter.y);
            ctx.drawImage(this.images[this.player], objectCenter.x - objectSize.x / 2, objectCenter.y - objectSize.y / 2, objectSize.x, objectSize.y);
            ctx.restore();
        }
    }

    inClickBounds(clickPos, objectCenter, objectSize) {
        return Math.sqrt((clickPos.x - objectCenter.x) ** 2 + (clickPos.y - objectCenter.y) ** 2) < objectSize.x / 3;
    }

    onClick() {
        //this.player = (this.player + 1) % 5;
        setRoad(this.roadX, this.roadY, game.player)
        .then(data => {
            game.updateObject()
        }
        )
    }
}

class CatanWebClient extends ObjectCanvas {
    constructor(ctx) {
        super(ctx);
        this.hexes = [];
        this.corners = [];
        this.roads = [];
        this.setupBoard();
        this.player = 1;
    }

    setupBoard() {
        var background = new backgroundWater(0, 0, 1, 1);
        this.addObject(background);
        var map = new backgroundMap(0, 0, 1, 1);
        background.addChild(map);

        var roadSize = 0.02;
        var hexSize = 0.175;

        for (var i = 0; i < 5; i++) {
            for (var j = 0; j < 5; j++) {
                if (
                    (j == 4 && i <= 1) ||
                    (j == 3 && i == 0) ||
                    (j == 0 && i >= 3) ||
                    (j == 1 && i == 4)
                ) {
                    continue;
                }
                var HexX = 0.866 * (hexSize + roadSize) * (i-1) - 0.433 * (hexSize + roadSize) * j;
                var HexY = 0.75 * (hexSize + roadSize) * (j-2);
                var hex = new Hex(HexX, HexY, hexSize, hexSize, i, j);
                this.hexes.push(hex);
                map.addChild(hex);
            }
        }

        for (var j = 0; j < 11; j++) {
            for (var i = 0; i < 11; i++) {

                if (
                    (j%2 == 1 && i % 2 == 1) ||
                    (j == 0 && i >= 6) ||
                    (j == 1 && i >= 7) ||
                    (j == 2 && i >= 8) ||
                    (j == 3 && i >= 9) ||
                    (j == 4 && i >= 10) ||
                    (j == 6 && i <= 0) ||
                    (j == 7 && i <= 1) ||
                    (j == 8 && i <= 2) ||
                    (j == 9 && i <= 3) ||
                    (j == 10 && i <= 4) 

                )
                {
                    continue;
                }
                
                var roadX = 0.433 * (hexSize + roadSize) * (i-3) - 0.2165 * (hexSize + roadSize) * (j-1);
                var roadY = 0.375 * (hexSize + roadSize) * (j-5);
                var rotation = 0;
                if (j%2 == 0 && i % 2 == 0) {
                    rotation = -1.04719755
                }else if (j%2 == 0 && i % 2 == 1) {
                    rotation = 1.04719755
                }
                var road = new Road(roadX, roadY, 3*roadSize, 2.4*roadSize, i, j, rotation);
                this.roads.push(road);
                map.addChild(road);
            }
        }

        for (var j = 0; j < 6; j++) {
            for (var i = 0; i < 12; i++) {

                if (
                    (j == 0 && i >= 7) ||
                    (j == 1 && i >= 9) ||
                    (j == 2 && i >= 11) ||
                    (j == 3 && i <= 0) ||
                    (j == 4 && i <= 2) ||
                    (j == 5 && i <= 4)
                ) {
                    continue;
                }
                
                var cornerX = 0.433 * (hexSize + roadSize) * (i) - 0.433 * (hexSize + roadSize) * j - 0.866 * (hexSize + roadSize) * 1.5;
                var cornerY = 0.75 * (hexSize + roadSize) * (j) - 0.25 * (hexSize + roadSize) * (i%2) - 0.75 * (hexSize + roadSize) * 2.34;
                var corner = new Corner(cornerX, cornerY, 3*roadSize, 3*roadSize, i, j);
                this.corners.push(corner);
                map.addChild(corner);
                // var dot = new Square(0, 0, 0.1, 0.1, "#FF0000");
                // corner.addChild(dot);
            }
        }
    }

    update() {
        getRoadsData()
        .then(data => {
            for (var i = 0; i < this.roads.length; i++) {
                this.roads[i].player = data[i];
            }
        });

        getRoadAvailability(this.player)
        .then(data => {
            for (var i = 0; i < this.roads.length; i++) {
                if (data[i] == 1) {
                    this.roads[i].available = true;
                }else {
                    this.roads[i].available = false;
                }
            }
        });

        getSettlementData()
        .then(data => {
            for (var i = 0; i < this.corners.length; i++) {
                this.corners[i].player = data[i * 2];
                this.corners[i].level = data[i * 2 + 1];
            }
        });

        getSettlementAvailability(this.player)
        .then(data => {
            for (var i = 0; i < this.corners.length; i++) {
                if (data[i] == 1) {
                    this.corners[i].available = true;
                }else {
                    this.corners[i].available = false;
                }
            }
        });
    }
}


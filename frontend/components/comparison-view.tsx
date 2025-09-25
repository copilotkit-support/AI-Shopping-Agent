"use client"

import { ArrowLeft, Heart, ExternalLink, Star } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface Product {
  id: string
  name: string
  price: string
  image: string
  pros: string[]
  cons: string[]
  source: string
  rating: number
  reviews: number
  specs: {
    processor: string
    ram: string
    storage: string
    display: string
    battery: string
    weight: string
    ports: string
    os: string
  }
}

interface ComparisonViewProps {
  products: Product[]
  onExit: () => void
  wishlist: string[]
  onToggleWishlist: (productId: string) => void
}

const specLabels = {
  processor: "Processor",
  ram: "RAM",
  storage: "Storage",
  display: "Display",
  battery: "Battery Life",
  weight: "Weight",
  ports: "Ports",
  os: "Operating System",
}

export function ComparisonView({ products, onExit, wishlist, onToggleWishlist }: ComparisonViewProps) {
  return (
    <div className="flex-1 bg-[#F7F7F9] overflow-y-auto">
      {/* Header */}
      <div className="bg-white border-b border-[#D8D8E5] p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={onExit} className="border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Products
            </Button>
            <div>
              <h1 className="text-2xl font-semibold text-[#030507] font-['Roobert']">Product Comparison</h1>
              <p className="text-[#575758]">Comparing {products.length} products side by side</p>
            </div>
          </div>
        </div>
      </div>

      {/* Comparison Grid */}
      <div className="p-6">
        <div className="grid gap-6" style={{ gridTemplateColumns: `repeat(${products.length}, 1fr)` }}>
          {products.map((product) => (
            <div key={product.id} className="bg-white rounded-xl border border-[#D8D8E5] overflow-hidden">
              {/* Product Header */}
              <div className="p-6 border-b border-[#D8D8E5]">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <h3 className="font-semibold text-[#030507] text-lg mb-2 font-['Roobert']">{product.name}</h3>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-2xl font-bold text-[#030507] font-['Roobert']">{product.price}</span>
                      <Badge variant="outline" className="text-xs">
                        {product.source}
                      </Badge>
                    </div>
                  </div>
                  <button
                    onClick={() => onToggleWishlist(product.id)}
                    className={`p-2 rounded-full transition-colors ml-3 ${
                      wishlist.includes(product.id)
                        ? "bg-[#FFA254] text-white hover:bg-[#FF8C42]"
                        : "bg-[#F7F7F9] text-[#858589] hover:bg-[#E8E8EF] hover:text-[#575758]"
                    }`}
                  >
                    <Heart className={`w-4 h-4 ${wishlist.includes(product.id) ? "fill-current" : ""}`} />
                  </button>
                </div>

                {/* Product Image */}
                <div className="aspect-video bg-[#F7F7F9] rounded-lg mb-4 overflow-hidden">
                  <img
                    src={product.image || "/placeholder.svg"}
                    alt={product.name}
                    className="w-full h-full object-cover"
                  />
                </div>

                {/* Rating */}
                <div className="flex items-center gap-2 mb-4">
                  <div className="flex items-center">
                    {[...Array(5)].map((_, i) => (
                      <Star
                        key={i}
                        className={`w-4 h-4 ${
                          i < Math.floor(product.rating) ? "text-[#FFF388] fill-current" : "text-[#D8D8E5]"
                        }`}
                      />
                    ))}
                  </div>
                  <span className="text-sm text-[#575758]">
                    {product.rating} ({product.reviews} reviews)
                  </span>
                </div>

                <Button className="w-full bg-[#030507] hover:bg-[#575758] text-white">
                  View Product
                  <ExternalLink className="w-4 h-4 ml-2" />
                </Button>
              </div>

              {/* Specifications */}
              <div className="p-6">
                <h4 className="font-semibold text-[#030507] mb-4">Specifications</h4>
                <div className="space-y-3">
                  {Object.entries(product.specs).map(([key, value]) => (
                    <div key={key} className="flex flex-col">
                      <span className="text-xs font-medium text-[#575758] uppercase tracking-wide mb-1">
                        {specLabels[key as keyof typeof specLabels]}
                      </span>
                      <span className="text-sm text-[#030507]">{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Pros and Cons */}
              <div className="p-6 border-t border-[#D8D8E5]">
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <h5 className="text-sm font-medium text-[#1B606F] mb-2">Pros</h5>
                    <ul className="space-y-1">
                      {product.pros.map((pro, index) => (
                        <li key={index} className="text-sm text-[#575758] flex items-start gap-2">
                          <span className="text-[#1B606F] mt-1">•</span>
                          {pro}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h5 className="text-sm font-medium text-[#FFA254] mb-2">Cons</h5>
                    <ul className="space-y-1">
                      {product.cons.map((con, index) => (
                        <li key={index} className="text-sm text-[#575758] flex items-start gap-2">
                          <span className="text-[#FFA254] mt-1">•</span>
                          {con}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

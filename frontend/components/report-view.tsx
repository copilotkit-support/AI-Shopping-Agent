"use client"

import { ArrowLeft, FileText, TrendingUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2 } from "lucide-react"
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

interface ReportViewProps {
  products: Product[]
  onExit: () => void
  searchQuery: string
  report: any
  isLoading: boolean
}

export function ReportView({ products, onExit, searchQuery, report, isLoading }: ReportViewProps) {

  return (
    <div className="flex-1 bg-[#F7F7F9] overflow-y-auto">
      {/* Header */}
      {isLoading ? (
        <div className="min-h-screen flex items-center justify-center bg-[#F7F7F9]">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-[#86ECE4] animate-spin mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-[#030507] mb-2">Generating report...</h3>
            <p className="text-[#575758]">This may take a few seconds</p>
          </div>
        </div>
      ) : (
        <>
          <div className="bg-white border-b border-[#D8D8E5] p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button variant="outline" onClick={onExit} className="border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Products
                </Button>
                <div>
                  <h1 className="text-2xl font-semibold text-[#030507] font-['Roobert']">AI Analysis Report</h1>
                  <p className="text-[#575758]">Comprehensive analysis</p>
                  {/* <p className="text-[#575758]">Comprehensive analysis for "{searchQuery}"</p> */}
                </div>
              </div>
              <Badge variant="outline" className="bg-[#86ECE4] text-[#030507] border-[#86ECE4]">
                <FileText className="w-4 h-4 mr-1" />
                AI Generated
              </Badge>
            </div>
          </div>

          <div className="p-6">
            {/* AI Recommendations */}
            <Card className="bg-white border-[#D8D8E5]">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-[#030507]">
                  <TrendingUp className="w-5 h-5 text-[#1B606F]" />
                  AI Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Top Pick */}
                  <div className="group rounded-xl p-[1px] bg-gradient-to-br from-[#86ECCE] via-[#FFF388] to-[#FFD6A5]">
                    <div className="rounded-xl p-4 h-full shadow-sm transition-all group-hover:shadow-md group-hover:-translate-y-0.5 bg-gradient-to-br from-[#E9FFF4] via-[#FFFBE6] to-[#E9F2FF]">
                      <h4 className="font-semibold text-[#030507] mb-1">🏆 Top Pick</h4>
                      <div className="text-sm text-[#030507] font-semibold mb-1">{report?.top_pick?.name}</div>
                      <p className="text-sm text-[#575758]">{report?.top_pick?.summary}</p>
                    </div>
                  </div>

                  {/* Best Performance */}
                  <div className="group rounded-xl p-[1px] bg-gradient-to-br from-[#86ECE4] via-[#1B606F] to-[#B3E5FC]">
                    <div className="rounded-xl p-4 h-full shadow-sm transition-all group-hover:shadow-md group-hover:-translate-y-0.5 bg-gradient-to-br from-[#E6FCF8] via-[#E6F3FF] to-[#F0E9FF]">
                      <h4 className="font-semibold text-[#1B606F] mb-1">✅ Best For Performance</h4>
                      <div className="text-sm text-[#030507] font-semibold mb-1">{report?.best_performance?.name}</div>
                      <p className="text-sm text-[#575758]">{report?.best_performance?.summary}</p>
                    </div>
                  </div>

                  {/* Best Value */}
                  <div className="group rounded-xl p-[1px] bg-gradient-to-br from-[#FFD6A5] via-[#FFA254] to-[#FFC371]">
                    <div className="rounded-xl p-4 h-full shadow-sm transition-all group-hover:shadow-md group-hover:-translate-y-0.5 bg-gradient-to-br from-[#FFF1E6] via-[#FFE6F7] to-[#FFF8E1]">
                      <h4 className="font-semibold text-[#FFA254] mb-1">💰 Best Value</h4>
                      <div className="text-sm text-[#030507] font-semibold mb-1">{report?.best_value_for_money?.name}</div>
                      <p className="text-sm text-[#575758]">{report?.best_value_for_money?.summary}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Detailed Comparison */}
            <Card className="mt-4 bg-white border-[#D8D8E5]">
              <CardHeader>
                <CardTitle className="text-[#030507]">Detailed Product Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-[#D8D8E5]">
                        <th className="text-left p-3 text-[#030507] font-semibold">Title</th>
                        {Object.entries(report?.products_specifications[0]?.specifications).map(([key, value]: any) => (
                          <th className="text-left p-3 text-[#030507] font-semibold">{value.split(":")[0]}</th>
                        ))}

                      </tr>
                    </thead>
                    <tbody>
                      {report?.products_specifications?.map((product: any, index: number) => (
                        <tr key={product?.name} className={index % 2 === 0 ? "bg-[#F7F7F9]" : "bg-white"}>
                          <td className="p-3">
                            <div className="flex items-center gap-3">
                              {/* <img
                            src={product?.image_urls[0] || "/placeholder.svg"}
                            alt={product?.name}
                            className="w-12 h-12 rounded-lg object-cover"
                          /> */}
                              <div>
                                <div className="font-semibold text-[#030507]">{product?.name}</div>
                                <div className="text-sm text-[#575758]">amazon.com</div>
                              </div>
                            </div>
                          </td>
                          {Object.entries(product?.specifications).map(([key, value]: any) => (
                            <td className="p-3">
                              {value.split(":")[1]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

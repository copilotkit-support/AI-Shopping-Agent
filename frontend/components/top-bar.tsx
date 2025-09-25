"use client"
import { ChevronDown, Settings, CreditCard, LogOut } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface TopBarProps {
  title: string
}

export function TopBar({ title }: TopBarProps) {
  return (
    <div className="h-16 bg-white border-b border-[#D8D8E5] flex items-center justify-between px-6">
      <h1 className="text-xl font-semibold text-[#030507] font-['Roobert']">{title}</h1>

      <DropdownMenu>
        <DropdownMenuTrigger className="flex items-center gap-3 p-2 rounded-lg hover:bg-[#E8E8EF] transition-colors">
          <Avatar className="w-8 h-8">
            <AvatarImage src="/placeholder.svg?height=32&width=32" />
            <AvatarFallback className="bg-[#86ECE4] text-[#030507] text-sm font-semibold">JD</AvatarFallback>
          </Avatar>
          <div className="text-left">
            <div className="text-sm font-semibold text-[#030507]">John Doe</div>
            <Badge variant="outline" className="text-xs">
              FREE
            </Badge>
          </div>
          <ChevronDown className="w-4 h-4 text-[#575758]" />
        </DropdownMenuTrigger>

        <DropdownMenuContent align="end" className="w-60 bg-white border-[#D8D8E5]">
          <DropdownMenuItem className="flex items-center gap-3 p-3 hover:bg-[#E8E8EF]">
            <Settings className="w-5 h-5 text-[#030507]" />
            <span className="text-[#030507]">Settings</span>
          </DropdownMenuItem>
          <DropdownMenuItem className="flex items-center gap-3 p-3 hover:bg-[#E8E8EF]">
            <CreditCard className="w-5 h-5 text-[#030507]" />
            <span className="text-[#030507]">Billing & Subscription</span>
          </DropdownMenuItem>
          <DropdownMenuItem className="flex items-center gap-3 p-3 hover:bg-[#E8E8EF]">
            <LogOut className="w-5 h-5 text-[#030507]" />
            <span className="text-[#030507]">Sign out</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
